"""
backend/detection/ocr/text_locator.py

Classical-CV dynamic text-region locator for OCRead.

The printed dot-matrix text on the bottle cap ("3030 22:42" / "BB 02/25")
does not land in the same spot on every bottle, so a single static
box would miss it whenever the cap is rotated/placed differently
under the camera. This module searches a (looser) region of the frame for
the printed text and returns its bounding box, so the OCR crop always
tracks the actual text instead of a fixed rectangle.

No trained model involved. Two stages:

1. Blackhat transform (cv2.MORPH_BLACKHAT) instead of a flat brightness
   threshold. A flat "pixel below N is ink" threshold also picks up the
   cap's embossed/molded features (the diagonal reinforcement ribs, the
   recycling triangle, "PE/PA") because those catch shadow under angled
   light and can be nearly as dark as the actual ink. Blackhat only
   responds to features *smaller than its kernel* against a locally
   brighter surround, so it suppresses the broad, large-scale shading of
   the molded ribs while still picking out the small printed characters.
2. Connected components on the blackhat result, then proximity-based
   clustering (union-find): printed text is many small components sitting
   close together, so the cluster with the most total ink *area* (not
   simply every surviving pixel unioned together, and not just the most
   components - see _largest_proximity_cluster's docstring for why count
   alone was foolable) is taken as the text block. This is what keeps
   stray specks - antialiasing on a mold line edge, a dust fleck, or a
   partially-erased rib fragmenting into a chain of small leftover pieces
   - from dragging the bounding box out to somewhere the text isn't.

Tunable entirely through cfg["ocr"]["dynamic_roi"] (see config/default.yaml).
Returns None when nothing plausible is found, so the caller can fall back
to reading the full frame.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

from backend.utils.logger import get_logger
from backend.utils.roi import roi_to_pixels

log = get_logger(__name__)


def locate_text_region(frame: np.ndarray, dyn_cfg: dict, debug_dir: Optional[str] = None) -> Optional[dict]:
    """
    Find the printed-text block inside *frame*.

    Args:
        frame:   BGR frame (the full camera frame, not yet cropped to any ROI).
        dyn_cfg: the ocr.dynamic_roi config dict.
        debug_dir: if set, intermediate masks + the final bbox overlay are
                   written here (mirrors the debug_ocr/ convention used
                   elsewhere). Leave unset in production - this is for
                   tuning the thresholds against real bottles.

    Returns:
        {"bbox": [x1, y1, x2, y2], "score": float} in *frame* pixel
        coordinates, or None if no plausible text region was found.
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]

    # Each call gets its own timestamped subfolder rather than writing to
    # fixed filenames directly in debug_dir - otherwise every cycle
    # overwrites the previous one's images, which makes this useless for
    # reviewing a session where you're rotating a bottle through several
    # angles (you'd only ever see whatever cycle happened to run last).
    cycle_debug_dir = None
    if debug_dir:
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cycle_debug_dir = os.path.join(debug_dir, stamp)

    # Search within a coarse region rather than the whole frame - this is
    # deliberately looser than the old static roi (enough to cover every
    # position the cap can land in), but still narrows the search away
    # from background/conveyor clutter outside the product itself.
    search_roi = dyn_cfg.get("search_roi") or {"x_start": 0.0, "y_start": 0.0, "x_end": 1.0, "y_end": 1.0}
    sx1, sy1, sx2, sy2 = roi_to_pixels(search_roi, w, h)
    if sx2 <= sx1 or sy2 <= sy1:
        sx1, sy1, sx2, sy2 = 0, 0, w, h

    search = frame[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY) if search.ndim == 3 else search

    # Blackhat = closing(gray) - gray. Highlights small dark features
    # against a locally brighter background and largely cancels out broad,
    # slowly-varying shading (the molded ribs/triangle) because closing
    # "fills those in" along with the true background before subtracting.
    blackhat_kernel_size = int(dyn_cfg.get("blackhat_kernel_size", 25))
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, blackhat_kernel_size), max(1, blackhat_kernel_size))
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    blackhat_threshold = int(dyn_cfg.get("blackhat_threshold", 25))
    _, dark_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)

    # Erase long straight line segments (the cap's molded ribs) from the
    # ink mask before doing anything else with it. On a clean reference
    # photo, blackhat mostly cancels these out on its own (broad,
    # slowly-varying shading) - but under real production lighting the rib
    # edges can catch enough of a shadow to come through as solid, sharp
    # lines that blackhat treats exactly like ink. A rib line is long
    # (150-400+px observed) and dead straight; no single character or
    # separator glyph comes anywhere close to that length (~39px was the
    # largest measured). Hough line detection with a minimum length well
    # above any real character reliably picks out just the rib arms, even
    # where a rib happens to run directly across a character - some of
    # that character's ink is lost too in that exact overlap (there's no
    # way to tell rib-ink from character-ink where they're genuinely the
    # same pixels), but that's a small, local loss compared to the
    # alternative of the rib dominating the whole cluster selection.
    if dyn_cfg.get("reject_long_lines", True):
        dark_mask = _erase_long_lines(
            dark_mask,
            min_length=int(dyn_cfg.get("long_line_min_length_px", 90)),
            thickness=int(dyn_cfg.get("long_line_erase_thickness_px", 6)),
        )

    # Small closing to merge dot-matrix fragments belonging to the same
    # character - deliberately much smaller than the blackhat kernel so it
    # doesn't re-bridge the gap back to the molded ribs.
    close_w = int(dyn_cfg.get("morph_kernel_w", 9))
    close_h = int(dyn_cfg.get("morph_kernel_h", 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, close_w), max(1, close_h)))
    closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)

    min_area = float(dyn_cfg.get("min_component_area", 3))
    max_area = float(dyn_cfg.get("max_component_area", 2000))
    # Per-component width/height cap, separate from the area cap above. A
    # single component's area alone doesn't catch every kind of oversized
    # non-text shape: a sparse, branching blob (e.g. where the cap's two
    # molded ribs cross) can have a fairly small area for how large its
    # bounding box is, and a long straight rib-edge segment can be thin
    # enough to also have a small area despite spanning most of the frame
    # in one dimension. Both slip through an area-only filter. Measured
    # against real dot-matrix characters on an actual cap, the largest
    # single character component was ~39x35px - this cap gives that
    # roughly 2x headroom for different bottles/lighting while still
    # rejecting rib/embossing fragments, which run 80-400+px in at least
    # one dimension.
    max_component_dim = float(dyn_cfg.get("max_component_dim", 70))

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    boxes = []
    label_ids = []
    areas = []
    for i in range(1, num_labels):  # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw > max_component_dim or ch > max_component_dim:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        boxes.append((x, y, x + cw, y + ch))
        label_ids.append(i)
        areas.append(int(area))

    if debug_dir:
        _write_debug(cycle_debug_dir, search=search, blackhat=blackhat, dark_mask=dark_mask, closed=closed)

    min_components = int(dyn_cfg.get("min_cluster_components", 4))
    # First pass: tight, isotropic clustering. This is deliberately
    # conservative (small gap) so it never bridges to unrelated debris -
    # it's just finding a reliable "seed" that's definitely part of the
    # real text.
    seed_gap = float(dyn_cfg.get("cluster_max_gap_px", 25))
    cluster = _largest_proximity_cluster(boxes, areas, max_gap=seed_gap)

    if not cluster or len(cluster) < min_components:
        log.debug(
            "text_locator: no plausible text cluster (total_components=%s best_cluster=%s, need>=%s)",
            len(boxes), len(cluster) if cluster else 0, min_components,
        )
        return None

    # Second pass: extend the seed along its own line direction, not just
    # isotropically. A rotated line of text has characters that are normal
    # spacing apart *along the baseline* but end up far apart in raw x/y
    # terms, since that spacing splits across both axes - the seed gap
    # above has to stay tight to avoid bridging to nearby noise (a rib
    # fragment, embossed lettering), which means it can under-cluster a
    # rotated line and clip off the tail end of it. Fitting a line through
    # the seed and only pulling in components that are both nearly on that
    # line (small perpendicular distance) and a plausible continuation of
    # it (not off in some unrelated direction) recovers the rest of the
    # line without reopening the door to unrelated clutter, since that
    # clutter essentially never sits precisely on the text's own line.
    cluster = _extend_cluster_along_line(
        boxes,
        cluster,
        max_perp_dist=float(dyn_cfg.get("line_extend_max_perp_px", 18)),
        max_along_gap=float(dyn_cfg.get("line_extend_max_gap_px", 90)),
    )

    xs1 = min(boxes[i][0] for i in cluster)
    ys1 = min(boxes[i][1] for i in cluster)
    xs2 = max(boxes[i][2] for i in cluster)
    ys2 = max(boxes[i][3] for i in cluster)

    padding = int(dyn_cfg.get("padding_px", 10))
    xs1 = max(0, xs1 - padding)
    ys1 = max(0, ys1 - padding)
    xs2 = min(search.shape[1], xs2 + padding)
    ys2 = min(search.shape[0], ys2 + padding)

    # Translate from search-crop coordinates back to full-frame coordinates.
    bbox = [sx1 + xs1, sy1 + ys1, sx1 + xs2, sy1 + ys2]

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    # Rotated (minimum-area) rect around the cluster, in addition to the
    # axis-aligned bbox above. The axis-aligned box only tells the caller
    # *where* the text is - it doesn't help when the cap (and therefore the
    # printed line) is rotated under the camera, since the crop still hands
    # OCR a diagonal line of text instead of a horizontal one. The rotated
    # rect captures the actual orientation of the printed line so the
    # caller can deskew it before running OCR - see deskew_crop() below.
    #
    # Fit to the actual ink pixels of the winning cluster, NOT to each
    # component's axis-aligned box corners. A rotated blob's axis-aligned
    # box over-estimates its true footprint by up to ~2x right around 45
    # degrees (and not at all at 0/90/180/270, where "rotated" and
    # "axis-aligned" coincide) - averaged over a whole cluster of
    # characters, that inflation was biasing/destabilizing the fitted
    # rectangle specifically at the angles between the cardinal points.
    # Fitting to real pixels removes that bias entirely, at any angle.
    cluster_label_ids = np.array([label_ids[i] for i in cluster])
    sub_labels = labels[ys1:ys2, xs1:xs2]
    ink_mask = np.isin(sub_labels, cluster_label_ids).astype(np.uint8)
    ink_points = cv2.findNonZero(ink_mask)

    if ink_points is not None and len(ink_points) >= 3:
        points = ink_points.reshape(-1, 2).astype(np.float32)
        points[:, 0] += xs1
        points[:, 1] += ys1
    else:
        # Degenerate fallback (shouldn't normally happen - the cluster
        # already passed the min_cluster_components check) - AABB corners
        # are still a valid, if less precise, rect.
        points = np.array(
            [[boxes[i][0], boxes[i][1]] for i in cluster]
            + [[boxes[i][2], boxes[i][1]] for i in cluster]
            + [[boxes[i][0], boxes[i][3]] for i in cluster]
            + [[boxes[i][2], boxes[i][3]] for i in cluster],
            dtype=np.float32,
        )
    (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(points)
    rw, rh, angle = _normalize_rect_angle(rw, rh, angle)

    rw += 2 * padding
    rh += 2 * padding

    rotated_rect = {
        "center": [sx1 + float(rcx), sy1 + float(rcy)],
        "size": [float(rw), float(rh)],
        "angle": float(angle),
    }

    # Simple confidence proxy: how many components formed the winning
    # cluster relative to a "plenty of characters" reference count.
    score = min(1.0, len(cluster) / 20.0)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.debug(
        "text_locator: found bbox=%s rotated_rect=%s cluster_components=%s score=%.3f (%sms)",
        bbox, rotated_rect, len(cluster), score, duration_ms,
    )

    if debug_dir:
        annotated = frame.copy()
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        box_pts = cv2.boxPoints(((rotated_rect["center"][0], rotated_rect["center"][1]), (rw, rh), angle))
        cv2.drawContours(annotated, [np.intp(box_pts)], 0, (0, 255, 0), 2)
        _write_debug(cycle_debug_dir, bbox_overlay=annotated)

    return {"bbox": bbox, "rotated_rect": rotated_rect, "score": score, "cycle_debug_dir": cycle_debug_dir}


def _normalize_rect_angle(rw: float, rh: float, angle: float) -> tuple[float, float, float]:
    """
    Normalize a cv2.minAreaRect() (width, height, angle) so that:

      1. the box is "landscape" (width = baseline direction), and
      2. the angle is the *smallest* rotation that levels the line,
         i.e. it lies in (-90, 90].

    (1) matters because minAreaRect doesn't know which side is the text
    baseline - without it a near-vertical line would keep its short side as
    width and deskew_crop() would rotate it 90 degrees the wrong way.

    (2) is a bug fix. Rotating by `angle` or by `angle - 180` both make the
    line horizontal; they differ only in whether the text ends up upright
    or upside-down. OpenCV reports minAreaRect angles in (0, 90], so after
    the +90 landscape swap the angle used to land anywhere in (0, 180].
    That meant every line tilted counter-clockwise - even by 2 degrees -
    was "levelled" by turning it ~178 degrees, i.e. handed to OCR upside
    down, while the same tilt clockwise came out upright. (Below
    deskew_crop's 1.5 degree snap tolerance no rotation happens at all, so
    the result also flipped back and forth around that threshold.) Taking
    the smaller of the two turns means any line within +/-90 degrees of
    upright always comes out upright; only text that is genuinely more
    than 90 degrees rotated comes out flipped, which the font reader
    handles by trying both orientations (font_reader.py).
    """
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    # Bring into (-90, 90]. A single step is enough for OpenCV >= 4.5.1
    # (angle in (0, 90] + optional 90), but loop anyway so older/newer
    # angle conventions can't silently reintroduce the flip.
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return rw, rh, angle


def deskew_crop(frame: np.ndarray, rotated_rect: dict, extra_padding_px: int = 0) -> Optional[np.ndarray]:
    """
    Straighten and crop *frame* to the rotated rect returned by
    locate_text_region(), so the printed line comes out horizontal
    regardless of how the cap was rotated under the camera.

    Fast path: if the rect is already axis-aligned to within
    ANGLE_SNAP_TOLERANCE_DEG (the common case on a line where bottles land
    close to upright, not rotated by a large angle every cycle), this skips
    rotation entirely and takes a direct pixel slice instead. warpAffine
    always resamples the image, even for a 1-degree rotation, which softens
    fine dot-matrix stroke edges for no real benefit when there's nothing
    meaningful to straighten - the fast path keeps those frames exactly as
    sharp as a plain static-ROI crop.

    For everything else: crop a generous axis-aligned margin around the
    rotated rect first (cheap), rotate only that small sub-image (not the
    full frame), then take the final tight crop out of the rotated
    sub-image. Rotating the whole frame for every read would work too but
    costs much more for no benefit, since we only ever want the pixels near
    the rect.

    Note on orientation: minAreaRect's angle is only defined up to 180
    degrees. locate_text_region() resolves that by always taking the
    smaller rotation (see _normalize_rect_angle), so text within +/-90
    degrees of upright always comes out upright. Text rotated by more than
    that (a cap that is genuinely upside-down under the camera) still
    comes out upside-down - that residual case is handled by the font
    reader, which tries both orientations (font_reader.py). The stages are
    complementary: this function handles "which way is the text tilted",
    the font reader handles "is it upside down".

    Returns None if the rect is degenerate or falls entirely outside frame.
    """
    ANGLE_SNAP_TOLERANCE_DEG = 1.5

    h, w = frame.shape[:2]
    cx, cy = rotated_rect["center"]
    rw, rh = rotated_rect["size"]
    angle = rotated_rect["angle"]

    rw += 2 * extra_padding_px
    rh += 2 * extra_padding_px

    if rw <= 1 or rh <= 1:
        return None

    angle_from_axis = min(abs(angle) % 180, 180 - (abs(angle) % 180))
    if angle_from_axis <= ANGLE_SNAP_TOLERANCE_DEG:
        x1 = int(round(cx - rw / 2))
        y1 = int(round(cy - rh / 2))
        x2 = int(round(cx + rw / 2))
        y2 = int(round(cy + rh / 2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # Axis-aligned margin big enough to contain the rotated rect at any
    # angle: half the rect's diagonal in every direction from its center.
    diag = int(np.ceil(((rw ** 2 + rh ** 2) ** 0.5) / 2)) + 2
    mx1, my1 = int(cx - diag), int(cy - diag)
    mx2, my2 = int(cx + diag), int(cy + diag)
    mx1c, my1c = max(0, mx1), max(0, my1)
    mx2c, my2c = min(w, mx2), min(h, my2)

    if mx2c <= mx1c or my2c <= my1c:
        return None

    margin = frame[my1c:my2c, mx1c:mx2c]

    # Center relative to the margin crop (not the full frame) now that
    # we've clamped to frame bounds.
    local_cx = cx - mx1c
    local_cy = cy - my1c

    mh, mw = margin.shape[:2]
    M = cv2.getRotationMatrix2D((local_cx, local_cy), angle, 1.0)
    rotated = cv2.warpAffine(margin, M, (mw, mh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    x1 = int(round(local_cx - rw / 2))
    y1 = int(round(local_cy - rh / 2))
    x2 = int(round(local_cx + rw / 2))
    y2 = int(round(local_cy + rh / 2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mw, x2), min(mh, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return rotated[y1:y2, x1:x2]


def _extend_cluster_along_line(boxes: list, seed: list, max_perp_dist: float, max_along_gap: float) -> list:
    """
    Grow a seed cluster by pulling in components that lie close to the
    seed's own line direction, even if they're farther away than the tight
    seed-clustering gap would normally allow.

    Fits a line through the seed's box centroids via PCA (robust to any
    angle, including near-vertical, unlike a plain x-on-y or y-on-x fit),
    then includes any other component whose centroid is within
    max_perp_dist of that line AND within max_along_gap of the seed's own
    extent measured along the line. Both conditions matter: perpendicular
    distance alone would still accept something far past the end of the
    real text if it happened to be collinear by coincidence, and the along
    gap alone would accept something off to the side. Requiring both is
    what lets this stay generous along the text's own direction while
    staying strict in every other direction.
    """
    if len(seed) < 2:
        return seed

    seed_set = set(seed)
    centroids = np.array([[(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2] for i in seed])
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]  # principal axis unit vector

    def project(pt):
        d = pt - mean
        along = float(np.dot(d, direction))
        perp = float(np.linalg.norm(d - along * direction))
        return along, perp

    seed_alongs = [project(c)[0] for c in centroids]
    lo, hi = min(seed_alongs), max(seed_alongs)

    extended = list(seed)
    for i, box in enumerate(boxes):
        if i in seed_set:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        along, perp = project(np.array([cx, cy]))
        if perp > max_perp_dist:
            continue
        if lo - max_along_gap <= along <= hi + max_along_gap:
            extended.append(i)

    return extended


def _largest_proximity_cluster(boxes: list, areas: list, max_gap: float) -> list:
    """
    Groups boxes via union-find, connecting any two whose gap (edge-to-edge
    distance) is <= max_gap, and returns the indices of the group with the
    most total ink area.

    Originally scored by component count instead of area - printed text is
    several small components sitting close together, so "the most members"
    seemed like a reasonable proxy for "the real text." In practice this
    was foolable: a partially-erased mold rib (see _erase_long_lines) can
    leave behind a sparse chain of small leftover fragments, and a chain
    like that can have MORE members than the real text block does despite
    being visually obvious debris - confirmed directly against a debug
    capture where this picked the embossed "PE/PA"/recycling-triangle
    region over the real ink two frames in three, because the broken rib
    fragments there simply out-counted the real characters.
    Total area is harder to fool the same way: real ink is a dense block
    of substantial character strokes, while a broken rib is thin and
    sparse even when it fragments into many pieces, so it takes a lot more
    pieces to match the same total area. A single unerased rib segment
    surviving as one big blob after the morphological close is still
    covered - it isn't picked because of its area, but because it isn't
    excluded either, it'd still need to pass the min_cluster_components
    check downstream, which a single component fails on its own the same
    way it always did.
    """
    n = len(boxes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def gap(b1, b2) -> float:
        x1a, y1a, x2a, y2a = b1
        x1b, y1b, x2b, y2b = b2
        dx = max(x1a - x2b, x1b - x2a, 0)
        dy = max(y1a - y2b, y1b - y2a, 0)
        if dx == 0 or dy == 0:
            return float(max(dx, dy))
        return float((dx ** 2 + dy ** 2) ** 0.5)

    for i in range(n):
        for j in range(i + 1, n):
            if gap(boxes[i], boxes[j]) <= max_gap:
                union(i, j)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    return max(clusters.values(), key=lambda idxs: sum(areas[i] for i in idxs))


def _erase_long_lines(mask: np.ndarray, min_length: int = 90, thickness: int = 6) -> np.ndarray:
    """Detect and blank out long straight line segments in a binary mask.

    Used to strip a bottle cap's molded ribs out of the ink mask before
    component detection - see the call site in locate_text_region() for
    why. Only erases segments at or above min_length, which should be set
    well above the longest plausible single character/glyph so real text
    is never at risk of being mistaken for a rib.

    Hough's own minLineLength parameter isn't just a post-hoc filter - it
    changes how it merges/detects segments in the first place, so a
    genuinely long rib broken up by antialiasing or mask noise could fail
    to be found as one segment at all if min_length were passed straight
    into the Hough call. It gets a low, permissive value instead so Hough
    can actually find candidate segments; min_length is enforced only in
    the filter loop below. (The same bug was found on a real crop in the
    former rib-inpainting step.)
    """
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=15)
    if lines is None:
        return mask
    out = mask.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length >= min_length:
            cv2.line(out, (x1, y1), (x2, y2), 0, thickness=thickness)
    return out


def _write_debug(debug_dir: str, **named_images) -> None:
    """Best-effort dump of intermediate images for threshold tuning."""
    try:
        os.makedirs(debug_dir, exist_ok=True)
        for name, img in named_images.items():
            cv2.imwrite(os.path.join(debug_dir, f"text_locator_{name}.png"), img)
    except Exception as exc:  # pragma: no cover - debug aid only, never fatal
        log.warning("text_locator debug write failed (non-fatal): %s", exc)