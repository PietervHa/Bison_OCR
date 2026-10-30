"""
backend/detection/ocr/dot_text.py

Cleans up a deskewed OCRead crop of dot-printed text before recognition, so
the recognizer sees dark text on a flat white background - with the cap's
molded ribs, embossing ("PE", recycling triangle) and lighting gradients
removed - instead of the raw plastic.

Why this replaced rib inpainting (the former derib step): on these caps the
rib's dark shadow edge runs straight through a printed character (the
trailing "6" of "4113 08:26" on the reference bottle). Derib found that edge
and inpainted a 7px band along it, which erased the character's stroke with
it - "4113 08:2" was the most common read in production (155 of 301 on
2026-09-03). Inpainting a band can never keep ink that sits on the rib.

What this does instead - separate print from plastic by *shape*:

1. Ink map: blackhat (closing minus image) picks out small dark features -
   the printed dots/strokes, but also the thin dark shadow edges of ribs.
2. Long-line removal: a grayscale opening with a 1px-wide line kernel keeps
   only structures at least `line_min_length_px` long in that direction.
   Taking the max over many directions gives "everything long and straight"
   - rib edges, mold lines - which is subtracted from the ink map. Printed
   strokes are far shorter than that, so they survive, including where they
   overlap a rib edge (only the edge's own darkness is subtracted there).
   Directions are sampled finely (every 5 degrees by default): with 15
   degree steps, a slightly tilted rib edge left a 1px remnant above and
   below the "6" that made it read as "5".
3. Local contrast normalisation: ink contrast varies across the curved cap
   (the "08:26" half of the reference print is much fainter), so each region
   is scaled against its own local maximum, with a floor so plain
   background noise isn't amplified into blobs.
4. Re-render as dark ink on white, optionally with a small blur
   (`bridge_sigma_px`) that merges separate dots into strokes - the software
   equivalent of slightly defocusing the camera, but applied only to the
   cleaned text, not to the ribs.

Measured on the reference frame (rotated through cap orientations, at two
camera scales, plus low-contrast and noisy variants), with a PP-OCRv4
recognizer as a stand-in for the production model: exact full reads went
from 3/44 (current pipeline, derib on) to 44/44, and 30/32 vs 1/32 on
held-out tilts. That is one bottle - validate on your own frames with
QC_tools/compare_ocr_preprocessing.py before trusting the numbers.

All pixel sizes are at camera resolution (the crop as cut from the frame),
before any upscaling.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    # Blackhat kernel: must be wider than a printed stroke/dot (so ink is
    # fully picked up) but narrower than broad shading.
    "ink_kernel_px": 19,
    # Straight structures at least this long are treated as plastic (rib
    # edges, mold lines) and removed. Must be comfortably longer than any
    # character stroke - including strokes of the two printed lines that
    # happen to be collinear (~2 character heights + line gap) - and shorter
    # than the visible length of a rib inside the crop.
    "line_min_length_px": 65,
    # Number of directions checked for long lines (180 / 36 = every 5 deg).
    "line_angles": 36,
    # Speed: find the long lines on the ink map scaled by this factor (max-
    # pooled, so a 1 px rib edge keeps its darkness; line_min_length_px
    # scaled along) and scale them back up, never above the ink itself.
    # The openings are ~85% of this function's time. 1.0 = full resolution.
    "line_scale": 1.0,
    # Speed: run the line openings on this many threads (backend/utils/
    # workers.py; OpenCV releases the GIL). The same result; 1 = serial.
    "threads": 1,
    # Window for local contrast normalisation, ~1 character height.
    "local_window_px": 21,
    # A region's reference contrast is never taken below this fraction of
    # the crop's overall ink level, so near-empty areas aren't amplified.
    "min_ref_frac": 0.4,
    # Ink weaker than this fraction of the local reference becomes white.
    "ink_floor": 0.1,
    # Gaussian sigma (camera px) used to merge separate dots into strokes.
    # 0 = off. The "slightly defocused camera reads better" effect, done in
    # software; tune on real sharp frames.
    "bridge_sigma_px": 0.0,
    # How ribs/lighting are removed from the ink:
    #   "divide"   - flat-field: image / estimated illumination (including
    #                rib shadows). Physically correct for shading: ink in a
    #                rib's shadow keeps its full contrast.
    #   "subtract" - ink map minus rib darkness (first version). Ink lying
    #                in a rib shadow loses part of its contrast.
    "illumination": "divide",
    # 0 = same small-crop rule as the classic preprocessing (usually 2x);
    # a fixed factor (e.g. 3) is used for extra-resolution reads.
    "upscale": 0,
}

_kernel_cache: dict = {}


def _line_kernels(length: int, n_angles: int) -> list:
    key = (length, n_angles)
    if key not in _kernel_cache:
        kernels = []
        c = (length - 1) / 2.0
        for i in range(n_angles):
            a = np.deg2rad(180.0 * i / n_angles)
            dx, dy = np.cos(a) * c, np.sin(a) * c
            k = np.zeros((length, length), np.uint8)
            cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
                     (int(round(c + dx)), int(round(c + dy))), 1, 1)
            kernels.append(k)
        _kernel_cache[key] = kernels
    return _kernel_cache[key]


def upscale_factor(short_side: int) -> int:
    """Small-crop upscale rule, kept from the former PaddleOCR preprocessing
    so the character size is the same as before."""
    if short_side >= 200:
        return 1
    return max(2, int(160 / max(1, short_side))) if short_side < 80 else 2


def clean_dot_text(crop: np.ndarray, cfg: dict | None = None, debug: dict | None = None) -> np.ndarray:
    """
    Return a BGR image of the crop's printed text as dark ink on white, with
    long straight structures (ribs, mold lines) and background removed.

    cfg: the ocr.dynamic_roi.dot_text config dict (missing keys use DEFAULTS).
    debug: if a dict is passed, intermediate float maps are stored in it
           ("ink", "lines", "text_ink") for inspection/tests.
    """
    p = dict(DEFAULTS)
    if cfg:
        p.update({key: v for key, v in cfg.items() if key in DEFAULTS and v is not None})

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray_f = gray.astype(np.float32)
    k = max(3, int(p["ink_kernel_px"]) | 1)
    # Background without anything small and dark: removes the print, but
    # also thin rib shadow lines (they're narrow too).
    background = cv2.morphologyEx(
        gray_f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    ink = background - gray_f  # == blackhat

    # Long straight dark structures (rib shadow edges, mold lines): grayscale
    # opening with 1px line kernels keeps only what is at least `length`
    # long in that direction. Print is far shorter, so it isn't in here.
    lines = np.zeros_like(ink)
    length = int(p["line_min_length_px"])
    scale_l = float(p["line_scale"] or 1.0)
    if length >= 3 and int(p["line_angles"]) > 0:
        if scale_l < 1.0:
            step = max(1, int(round(1.0 / scale_l)))
            small = cv2.dilate(ink, np.ones((step, step), np.uint8))[::step, ::step]
            small_lines = np.zeros_like(small)
            for lk in _line_kernels(max(3, int(round(length / step))), int(p["line_angles"])):
                np.maximum(small_lines, cv2.morphologyEx(small, cv2.MORPH_OPEN, lk), out=small_lines)
            lines = np.minimum(cv2.resize(small_lines, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_LINEAR),
                               ink)
        elif int(p["threads"] or 1) > 1:
            from backend.utils.workers import pool
            opened = pool(int(p["threads"])).map(lambda lk: cv2.morphologyEx(ink, cv2.MORPH_OPEN, lk),
                                                 _line_kernels(length, int(p["line_angles"])))
            for o in opened:
                np.maximum(lines, o, out=lines)
        else:
            for lk in _line_kernels(length, int(p["line_angles"])):
                np.maximum(lines, cv2.morphologyEx(ink, cv2.MORPH_OPEN, lk), out=lines)

    if str(p["illumination"]).lower() == "subtract":
        text_ink = np.clip(ink - lines, 0, None)
    else:
        # Illumination = background with the long shadow lines put back.
        # Dividing by it turns the image into relative reflectance (~1 for
        # plastic, ~0.3 for ink) independent of rib shadows and lighting
        # gradients; ink is the relative darkening. Scaled by the median
        # background so the thresholds below see the same range as before.
        illumination = np.maximum(background - lines, 1.0)
        reflectance = gray_f / illumination
        text_ink = np.clip(1.0 - reflectance, 0, None) * float(np.median(background))

    ceil = float(np.percentile(text_ink, 99.5))
    if ceil <= 0:
        norm = np.zeros_like(text_ink)
    else:
        win = max(3, int(p["local_window_px"]) | 1)
        local_max = cv2.dilate(text_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
        local_max = cv2.GaussianBlur(local_max, (0, 0), win / 4.0)
        ref = np.maximum(local_max, float(p["min_ref_frac"]) * ceil)
        floor = float(p["ink_floor"])
        norm = np.clip((text_ink / ref - floor) / max(1e-6, 1.0 - floor), 0, 1)

    scale = int(p["upscale"]) if int(p["upscale"] or 0) > 0 else upscale_factor(min(gray.shape[:2]))
    if scale != 1:
        norm = cv2.resize(norm, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    sigma = float(p["bridge_sigma_px"])
    if sigma > 0:
        norm = cv2.GaussianBlur(norm, (0, 0), sigma * scale)
        peak = float(np.percentile(norm, 99.5))
        if peak > 0:
            norm = norm / peak

    out = (255.0 - 225.0 * np.clip(norm, 0, 1)).astype(np.uint8)

    if debug is not None:
        debug.update(ink=ink, lines=lines, text_ink=text_ink)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
