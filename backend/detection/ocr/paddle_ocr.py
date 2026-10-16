"""
PaddleOCR Integration

Provides the PaddleOCR class for text recognition using the PaddleOCR engine.
"""

import cv2
import numpy as np
import re
import time
import logging
import os
import string
import unicodedata
from collections import Counter
from pathlib import Path
from backend.detection.ocr.paddle_worker import PaddleOCRWorker, PaddleWorkerError
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
from backend.utils.roi import apply_roi, draw_roi
from backend.detection.ocr.text_locator import locate_text_region, deskew_crop
from backend.detection.ocr.dot_text import clean_dot_text
from backend.detection.ocr.code_validator import CodeTemplate, REJECT_CONSENSUS, REJECT_FORMAT, REJECT_CONFIDENCE
from backend.detection.ocr.font_reader import FontReader

log = get_logger(__name__)

logging.getLogger("ppocr").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path) -> str:
    """Relative paths in the config are relative to the project root - the
    same as data/results and data/logs - NOT to whatever folder the app was
    started from. (debug_dir used to be taken relative to the working
    directory, so starting the app from backend/ silently wrote the debug
    captures to backend/data/debug/dynamic_roi.)"""
    p = Path(str(path))
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# Characters OCR output is allowed to contain unless ocr.allowed_chars
# overrides it - see _sanitize_candidates().
DEFAULT_ALLOWED_CHARS = string.ascii_letters + string.digits + " :/.-"


class PaddleOCR:
    """
    OCR engine based on PaddleOCR for identifying keywords and dates in frames.
    """
    def __init__(self, app_state=None):
        self.app_state = app_state
        ocr_cfg = cfg["ocr"]
        self._paddle = PaddleOCRWorker(
            lang="en",
            cpu_threads=int(ocr_cfg.get("cpu_threads", 4)),
            use_angle_cls=bool(ocr_cfg.get("use_angle_cls", True)),
            enable_mkldnn=bool(ocr_cfg.get("enable_mkldnn", True)),
            text_detection_model_name=ocr_cfg.get("text_detection_model_name") or None,
            text_recognition_model_name=ocr_cfg.get("text_recognition_model_name") or None,
            inference_timeout=float(ocr_cfg.get("inference_timeout_s", 10.0)),
        )
        self.keywords = [w.lower() for w in ocr_cfg["keywords"]]
        self.keyword_set = set(self.keywords)
        self.date_regex = ocr_cfg["date_regex"]
        self._date_pattern = re.compile(self.date_regex) if self.date_regex else None
        self.debug_draw_roi = cfg["hmi"]["debug_draw_roi"]
        self.preprocess_mode = ocr_cfg["preprocess"].lower()
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])

        # Dynamic ROI (OCRead mode only): locates the printed text block via
        # classical CV instead of relying on the static cfg["roi"] box, since
        # the text doesn't land in the same place on every bottle. See
        # text_locator.py. Falls back to the static ROI when disabled or
        # when nothing is found on a given frame.
        dynamic_roi_cfg = ocr_cfg.get("dynamic_roi", {}) if isinstance(ocr_cfg.get("dynamic_roi"), dict) else {}
        self.dynamic_roi_enabled = bool(dynamic_roi_cfg.get("enabled", False))
        self.dynamic_roi_cfg = dynamic_roi_cfg
        self.dynamic_roi_debug_dir = (
            resolve_project_path(dynamic_roi_cfg.get("debug_dir") or "data/debug/dynamic_roi")
            if dynamic_roi_cfg.get("debug_save") else None
        )
        if self.dynamic_roi_debug_dir:
            log.info("OCRead debug captures are written to %s", self.dynamic_roi_debug_dir)

        # Rib inpainting (OCRead / dynamic ROI only): on this cap, the
        # mold rib crosses directly through the same 1-2 leading
        # characters of the printed line on almost every cycle, because
        # the rib position and the print position never move relative to
        # each other - confirmed against real debug captures, not a
        # hypothetical. text_locator.py already detects+erases rib lines,
        # but only inside the *clustering* mask used to locate the text -
        # the rib pixels are still physically present in the crop that
        # actually gets handed to the recognizer. This re-detects long
        # straight lines directly on the deskewed crop and inpaints over
        # them before recognition, instead of just working around the rib
        # during localization.
        #
        # Off by default - validate against dynamic_roi.debug_save
        # captures on your own frames before trusting this in production.
        # See _derib_crop() for why the thresholds need checking against
        # real crops rather than assumed safe.
        self.derib_enabled = bool(dynamic_roi_cfg.get("derib_enabled", False))
        self.derib_blackhat_kernel = int(dynamic_roi_cfg.get("derib_blackhat_kernel", 25))
        self.derib_threshold = int(dynamic_roi_cfg.get("derib_threshold", 32))
        self.derib_min_length_frac = float(dynamic_roi_cfg.get("derib_min_length_frac", 0.6))
        self.derib_thickness = int(dynamic_roi_cfg.get("derib_thickness", 7))

        # Dot-text cleanup (OCRead / dynamic ROI only) - see dot_text.py.
        # Removes ribs/mold lines by shape instead of inpainting over them,
        # so ink printed on a rib survives, and re-renders the print as dark
        # text on white. When enabled it replaces both _derib_crop() (which
        # erased the stroke of a character sitting on a rib edge) and the
        # CLAHE step of _preprocess_image() for OCRead crops.
        dot_text_cfg = dynamic_roi_cfg.get("dot_text") if isinstance(dynamic_roi_cfg.get("dot_text"), dict) else {}
        self.dot_text_enabled = bool(dot_text_cfg.get("enabled", False))
        self.dot_text_cfg = dot_text_cfg
        if self.dot_text_enabled and self.derib_enabled:
            log.info("ocr.dynamic_roi.dot_text is enabled, so derib_enabled is ignored for OCRead crops.")

        # Read validation (OCRead) - see code_validator.py. When enabled, a
        # read is only reported OK if it fits the configured print template
        # with enough confidence and (consensus) enough independently
        # preprocessed reads agree; otherwise NOK with a reason instead of
        # forwarding a guess to the PLC.
        val_cfg = ocr_cfg.get("read_validation") if isinstance(ocr_cfg.get("read_validation"), dict) else {}
        self.validator = None
        self.consensus_variants = [None]
        self.consensus_min_agree = 1
        self.consensus_fusion = "exact"
        self.consensus_early_stop = True
        self.consensus_min_reads = 2
        self.consensus_min_share = 0.7
        self.region_margin_frac = 0.6
        self.reader = "paddle"
        self.font_reader = None
        self._last_located = None
        if val_cfg.get("enabled", False):
            self.validator = CodeTemplate.from_config(val_cfg)
            cons = val_cfg.get("consensus") if isinstance(val_cfg.get("consensus"), dict) else {}
            variants = [str(v) for v in (cons.get("variants") or [])] or [None]
            unknown = [v for v in variants if v is not None and v not in self.PREPROCESS_VARIANTS]
            if unknown:
                raise ValueError(f"ocr.read_validation.consensus.variants: unknown {unknown}; "
                                 f"choose from {sorted(self.PREPROCESS_VARIANTS)}")
            self.consensus_variants = variants
            self.consensus_min_agree = max(1, min(int(cons.get("min_agree", 1)), len(variants)))
            # "exact": OK when min_agree reads give the identical code (stops early).
            # "position_vote": every read that fits the template votes per
            # character position (weighted by its confidence); OK only when
            # >= min_reads reads fit and every position's winner holds
            # >= min_position_share of the weight. Reads all variants.
            self.consensus_fusion = str(cons.get("fusion", "exact"))
            if self.consensus_fusion not in ("exact", "position_vote"):
                raise ValueError("ocr.read_validation.consensus.fusion must be 'exact' or 'position_vote'")
            # early_stop (exact fusion only): stop as soon as min_agree reads
            # agree (fast). false = read every variant and also require the
            # winning code to beat every other code (no ties) - slower, but
            # it is the rule the accuracy numbers in default.yaml were measured with.
            self.consensus_early_stop = bool(cons.get("early_stop", True))
            self.consensus_min_reads = int(cons.get("min_reads", 2))
            self.consensus_min_share = float(cons.get("min_position_share", 0.7))
            self.region_margin_frac = float(cons.get("region_margin_frac", 0.6))

            # Reader: "font" = classifier trained on this printer's dot font
            # (font_reader.py); "paddle" = PaddleOCR consensus as above.
            self.reader = str(val_cfg.get("reader", "font")).lower()
            fr_cfg = val_cfg.get("font_reader") if isinstance(val_cfg.get("font_reader"), dict) else {}
            if self.reader == "font":
                try:
                    model = fr_cfg.get("model") or None
                    self.font_reader = FontReader(
                        resolve_project_path(model) if model else None,
                        min_literal_score=float(fr_cfg.get("min_literal_score", -1.0)),
                        min_digit_confidence=float(fr_cfg.get("min_digit_confidence", 0.0)),
                    )
                    if not self.font_reader.supports(self.validator.lines) or len(self.validator.lines) != 2:
                        log.warning("Font reader does not support template %s - using PaddleOCR reader.",
                                    self.validator.lines)
                        self.font_reader = None
                except Exception as exc:
                    log.error("Font reader could not be loaded (%s) - using PaddleOCR reader.", exc)
                    self.font_reader = None
            log.info("OCRead reader: %s", "font" if self.font_reader is not None else "paddle")
            log.info("OCRead validation: template=%s variants=%s min_agree=%d min_confidence=%.2f",
                     self.validator.lines, variants, self.consensus_min_agree, self.validator.min_confidence)

        # Orientation retry (OCRead / dynamic ROI only): deskew_crop()
        # resolves *which way* a rotated line is tilted, but not whether
        # it came out upside-down (0 vs 180 degrees) - that ambiguity is
        # left to PaddleOCR's own angle classifier (use_angle_cls). On
        # this print - tiny, sparse dot-matrix digits - that classifier
        # isn't reliable: debug captures show visibly worse reads on
        # cycles where the cap was genuinely upside-down in frame. Rather
        # than trust it blindly every cycle, a low-confidence (or empty)
        # first pass is retried against a 180-degree-rotated copy of the
        # same crop, keeping whichever pass scores higher. Only fires when
        # the first pass looks doubtful, so the common case (upright,
        # confident read) pays no extra latency.
        self.orientation_retry_enabled = bool(dynamic_roi_cfg.get("orientation_retry_enabled", True))
        self.orientation_retry_confidence_threshold = float(
            dynamic_roi_cfg.get("orientation_retry_confidence_threshold", 0.85)
        )
        # Whether low confidence alone (not just an empty result) should
        # trigger the retry - see the long comment above
        # orientation_retry_confidence_threshold in default.yaml. Defaults
        # False now: three threshold values (0.85, 0.6, 0.8) were each
        # tried against real capture data, and none of them cleanly
        # separated genuine upside-down cycles from a persistent,
        # orientation-independent recognition ambiguity on this cap's
        # leading digit - worse, when the retry fired on the latter, it
        # picked the wrong (flipped) answer 13/13 times in one 30-cycle
        # capture. Retrying on a fully empty result is still safe and
        # still on by default; retrying on "found something, just not
        # very confident" is not, until that ambiguity is actually
        # understood rather than threshold-tuned around.
        self.orientation_retry_require_low_confidence = bool(
            dynamic_roi_cfg.get("orientation_retry_require_low_confidence", False)
        )

        # Any cycle slower than this gets its full stage breakdown
        # (roi_ms/derib_ms/ocr_ms/orientation_retry_ms/total_ocr_call_ms)
        # logged as a WARNING, regardless of whether the caller passed
        # profile=True - see the auto-log block in _recognize(). This is
        # what previous multi-second spikes were missing: elapsed_ms alone
        # doesn't say whether the time went into Paddle's own compute or
        # somewhere it shouldn't have.
        self.slow_cycle_log_threshold_ms = float(ocr_cfg.get("slow_cycle_log_threshold_ms", 800))

        # Output character filter - see _sanitize_candidates(). Once
        # text_detection/recognition_model_name are pinned, PaddleOCR
        # ignores lang="en" and the recognizer uses its full multilingual
        # dictionary, so noise on the cap regularly comes back as CJK
        # characters (seen in the results DB: 心, 门, 一, 日). Those were
        # then sent to the PLC as multi-byte UTF-8. An empty/null
        # ocr.allowed_chars disables the filter.
        allowed_chars = ocr_cfg.get("allowed_chars", DEFAULT_ALLOWED_CHARS)
        self._allowed_chars = frozenset(str(allowed_chars)) if allowed_chars else None

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        log.debug(
            "PaddleOCR init: preprocess_mode=%s downscale=%s keywords=%s derib_enabled=%s "
            "orientation_retry_enabled=%s dot_text_enabled=%s",
            self.preprocess_mode,
            self.downscale,
            self.keywords,
            self.derib_enabled,
            self.orientation_retry_enabled,
            self.dot_text_enabled,
        )

    def warmup(self) -> None:
        """Runs one throwaway OCR pass so Paddle's native-runtime cold-start
        cost is paid at startup instead of during the first real OCR cycle."""
        try:
            import numpy as np
            self._paddle.ocr(np.zeros((64, 64, 3), dtype="uint8"))
            log.debug("PaddleOCR warm-up inference completed.")
        except Exception as exc:
            log.warning("PaddleOCR warm-up inference failed (non-fatal): %s", exc)

    def _preprocess_image(self, frame):
        """Enhance the ROI crop for PaddleOCR's recognizer before inference.

        This is deliberately NOT the same treatment as tesseract_ocr.py's
        version. PaddleOCR's recognition model is a CNN trained on natural,
        anti-aliased grayscale/color crops - hard binarization (Otsu/
        adaptive threshold, which Tesseract benefits from) throws away the
        soft edge gradients that CNN was trained on and tends to hurt
        accuracy rather than help it. So this only ever does two things:
        upscale small crops, and gently lift local contrast. It never
        produces a binary mask.

        Upscaling matters a lot here specifically because of dynamic ROI:
        it now hands this function a tight crop of just the printed line(s)
        instead of the old large static box, so a two-line dot-matrix stamp
        can end up as little as ~25-30px tall per line - well below the
        ~40-60px recognition accuracy tends to need. mode == "off" skips
        all of this and is intended for debugging/comparison, not normal
        production use.
        """
        mode = self.preprocess_mode
        if mode == "off":
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        h, w = gray.shape[:2]
        short = min(h, w)
        if short < 200:
            # Same upscale heuristic as tesseract_ocr.py, for consistency -
            # small crops get a stronger boost, capping out around 2-3x
            # rather than scaling indefinitely.
            scale = max(2, int(160 / short)) if short < 80 else 2
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        enhanced = self._clahe.apply(gray)

        if mode == "accurate":
            enhanced = cv2.medianBlur(enhanced, 3)

        # Paddle's recognizer expects a 3-channel image - replicate the
        # enhanced grayscale back out rather than passing single-channel,
        # since the print here is essentially monochrome ink on plastic
        # anyway and this avoids any channel-count surprises in the model.
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _derib_crop(self, crop):
        """
        Detects and inpaints long straight rib lines crossing the deskewed
        OCRead crop, before the recognizer ever sees it.

        Same detection technique as text_locator._erase_long_lines()
        (blackhat -> threshold -> HoughLinesP), but run directly on the
        crop and used to inpaint real pixels rather than blank a
        clustering mask - see the comment above self.derib_enabled in
        __init__ for why this exists.

        min_length_frac is relative to the crop's own largest dimension
        rather than an absolute pixel count, since dynamic ROI crops vary
        in size cycle to cycle (unlike text_locator's search crop, which
        is close to full-frame-sized every time). Keep this conservative:
        a crop is fit tightly around the text, so the *real* printed line
        can itself span a large fraction of the crop's width, and in
        principle a row of characters bridged by HoughLinesP's maxLineGap
        could be mistaken for one long line. In practice individual
        digit/letter shapes rarely present one straight pixel-level edge
        the way a rib does, but verify against dynamic_roi.debug_save
        captures (compare the pre/post image) before trusting this on the
        line, and tighten min_length_frac or thickness if it ever bites
        into real characters instead of the rib.

        Best-effort: any failure or "nothing found" returns the crop
        unchanged, since a missed rib is no worse than today's behaviour.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            h, w = gray.shape[:2]

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(1, self.derib_blackhat_kernel),) * 2
            )
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, self.derib_threshold, 255, cv2.THRESH_BINARY)

            min_length = int(max(w, h) * self.derib_min_length_frac)
            # Hough's own minLineLength isn't just a post-hoc filter - it
            # changes how Hough merges/detects segments in the first place.
            # Passing our real target length directly into it here meant a
            # genuinely-long rib that's slightly broken (antialiasing, a
            # touch of noise in the blackhat mask) could fail to be found
            # as one segment at all, well before ever reaching the length
            # check below - confirmed directly: a real 135px rib on a real
            # crop went completely undetected at min_length_frac=0.6
            # (169px required) AND still failed with minLineLength passed
            # in directly even after lowering the fraction, only starting
            # to work once Hough's own minLineLength was decoupled from
            # the length filter. Hough gets a low, permissive value so it
            # can actually find candidate segments; the real length
            # requirement is enforced only in the filter loop below.
            lines = cv2.HoughLinesP(
                mask, 1, np.pi / 180,
                threshold=40, minLineLength=30, maxLineGap=10,
            )
            if lines is None:
                return crop

            inpaint_mask = np.zeros((h, w), dtype=np.uint8)
            found = False
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < min_length:
                    continue
                cv2.line(inpaint_mask, (x1, y1), (x2, y2), 255, thickness=self.derib_thickness)
                found = True

            if not found:
                return crop

            return cv2.inpaint(crop, inpaint_mask, 3, cv2.INPAINT_TELEA)
        except Exception as exc:
            log.warning("Rib inpainting failed (non-fatal), crop used as-is: %s", exc)
            return crop

    def _downscale_roi(self, frame):
        if self.downscale >= 1.0:
            return frame
        h, w = frame.shape[:2]
        if self.min_dim and min(h, w) <= self.min_dim:
            return frame
        scale = self.downscale
        if self.min_dim:
            scale = max(scale, self.min_dim / float(min(h, w)))
        if scale >= 1.0:
            return frame
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _apply_roi(self, frame, dynamic=False):
        """Crop the frame to the text region.

        dynamic=True is the OCRead-mode path, and it never touches
        cfg["roi"]. It only ever does one of two things: crop to wherever
        locate_text_region() actually found ink on this frame (deskewed via
        deskew_crop), or - if it found nothing at all - hand the full,
        uncropped frame to OCR rather than guessing a fixed rectangle. A
        production line where bottles don't land the same way every time
        means a static box is wrong as often as it's right, so there is no
        static-ROI fallback in this path at all. No debug overlay is drawn
        here either - the detected region changes on every frame, so a box
        drawn on the live view wouldn't mean anything to line staff.

        dynamic=False (the plain "ocr" keyword-match mode) is unrelated to
        OCRead and still uses the static cfg["roi"] box, via the shared roi
        utility so the coordinate math isn't duplicated across web.py /
        tesseract_ocr.py.

        Returns (crop, cycle_debug_dir). cycle_debug_dir is the same
        per-cycle folder text_locator.py just wrote its own debug images
        into (None if dynamic_roi debug_save is off, or in the static-ROI
        path, which has no equivalent) - _recognize() uses it to write the
        actual OCR result alongside the localization debug images, so a
        debug capture shows both where the crop came from and what OCR did
        with it, instead of just the former.
        """
        if dynamic:
            if not self.dynamic_roi_enabled:
                log.warning(
                    "OCRead called with dynamic ROI disabled (ocr.dynamic_roi.enabled=false) - "
                    "running OCR on the full frame. Enable it in config to locate text automatically."
                )
                self._last_located = None
                return frame, None

            located = locate_text_region(frame, self.dynamic_roi_cfg, debug_dir=self.dynamic_roi_debug_dir)
            # Kept for OCRead's frame-region reads (read() -> _frame_variant_input).
            self._last_located = located or None
            if not located:
                log.debug("Dynamic ROI: no text region found on this frame - running OCR on the full frame.")
                return frame, None

            cycle_debug_dir = located.get("cycle_debug_dir")
            rotated_rect = located.get("rotated_rect")
            # Extra margin here (beyond dynamic_roi.padding_px, which is
            # already baked into rotated_rect) is specifically for
            # PaddleOCR's own internal detector, which tends to need a bit
            # of non-text margin around a region to reliably fire - a crop
            # with characters right up against the edge is more likely to
            # be missed by its detector even when a human would read it
            # fine. This is separate from (and on top of) the tighter
            # padding used for our own CV clustering, which stays as-is
            # since loosening that risks pulling ribs back into detection.
            crop = deskew_crop(frame, rotated_rect, extra_padding_px=20) if rotated_rect else None
            if crop is None:
                # Degenerate rotated rect (shouldn't normally happen) - use
                # the axis-aligned box we already computed rather than the
                # full frame, since we did find *something*.
                x1, y1, x2, y2 = located["bbox"]
                crop = frame[y1:y2, x1:x2]

            return crop, cycle_debug_dir

        # Static-ROI path: only reached from run() (the "ocr" keyword mode).
        # read() (OCRead) always passes dynamic=True and never gets here.
        roi = cfg.get("roi")
        if not roi:
            return frame, None

        if self.debug_draw_roi:
            draw_roi(frame, roi)

        return apply_roi(frame, roi), None

    def _write_ocr_result_debug(self, debug_dir, display_frame, candidates, elapsed_ms, extra_lines=None,
                                name="ocr_result.png"):
        """Write the actual OCR outcome (not just the crop) into the same
        per-cycle debug folder text_locator.py used, so a debug capture
        shows what was read, not only where the crop came from - reviewing
        localization and recognition together instead of needing a second
        round-trip for log lines.

        extra_lines: optional list of extra "key=value" strings appended
        to the report header - currently used to record whether the
        orientation retry (see _recognize()) ended up using the flipped
        crop, so a debug capture also shows *why* a read looks the way it
        does, not just what it read.
        """
        try:
            full_text = " ".join(c["text"] for c in candidates).strip()
            header_lines = [
                f"full_text={full_text!r}",
                f"elapsed_ms={round(elapsed_ms, 1)}",
                f"candidate_count={len(candidates)}",
            ]
            if extra_lines:
                header_lines.extend(extra_lines)
            body_lines = [f"text={c['text']!r} confidence={c['confidence']}" for c in candidates]
            report = "\n".join(header_lines + body_lines) + "\n"

            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write(report)

            # Captioned image too, for a quick visual glance without having
            # to open the text file - shows exactly what PaddleOCR received
            # (post-preprocessing) plus what it read from it.
            img = display_frame
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img.shape[:2]
            caption_h = 30
            canvas = np.full((h + caption_h, max(w, 320), 3), 255, dtype=np.uint8)
            canvas[:h, :w] = img
            caption = full_text if full_text else "(no text detected)"
            cv2.putText(canvas, caption, (5, h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(debug_dir, name), canvas)
        except Exception as exc:  # pragma: no cover - debug aid only, never fatal
            log.warning("OCR result debug write failed (non-fatal): %s", exc)

    # Preprocessing variants usable for consensus reads (read_validation).
    # On the deskewed tight crop from the text locator:
    CROP_VARIANTS = {
        "dot",           # dot_text with its configured illumination model
        "dot_divide",    # dot_text, flat-field (divide)
        "dot_subtract",  # dot_text, subtractive
        "classic",       # grayscale + upscale + CLAHE (previous default, no derib)
        "classic_derib", # previous default with rib inpainting
        # Test-time variants of dot_divide/dot_subtract: same method, slightly
        # different input, so their errors are less correlated and a vote
        # between them means something.
        "dot_divide_x3", "dot_subtract_x3",          # 3x instead of 2x upscale
        "dot_divide_tilt_p3", "dot_divide_tilt_m3",  # crop rotated +/-3 degrees
        "dot_subtract_tilt_p3", "dot_subtract_tilt_m3",
        "dot_divide_bridge",                         # dots merged into strokes (sigma 0.7)
    }
    # On a larger part of the frame, letting PaddleOCR's own detector find
    # and straighten the lines - the way the plain OCR pipeline reads, which
    # detects dot text well but only inside its fixed box:
    FRAME_VARIANTS = {
        "region",        # generous box around the located text, OCR-pipeline preprocessing
        "region_dot",    # same box, dot_text (flat-field) preprocessing
        "full_frame",    # whole frame, OCR-pipeline preprocessing
        "ocr_pipeline",  # exactly what the OCR pipeline reads: static cfg["roi"] box + its preprocessing
    }
    PREPROCESS_VARIANTS = CROP_VARIANTS | FRAME_VARIANTS

    @staticmethod
    def _order_candidates(candidates):
        """Order tokens in reading order from their boxes: group into lines
        by vertical centre, top to bottom, then left to right inside a line.
        The model's own output order isn't guaranteed to be reading order
        (seen: "BB 4113 08:26 05/26"). Left unchanged if any box is missing."""
        if not candidates or not all(
            isinstance(c.get("box"), (list, tuple)) and len(c["box"]) >= 2 for c in candidates
        ):
            return candidates
        items = []
        for c in candidates:
            try:
                xs = [float(p[0]) for p in c["box"]]
                ys = [float(p[1]) for p in c["box"]]
            except (TypeError, ValueError, IndexError):
                return candidates
            items.append([(min(ys) + max(ys)) / 2.0, max(ys) - min(ys), min(xs), c])
        items.sort(key=lambda it: it[0])
        lines = []
        for it in items:
            if lines:
                line = lines[-1]
                centre = sum(i[0] for i in line) / len(line)
                height = max(max(i[1] for i in line), it[1], 1.0)
                if abs(it[0] - centre) < 0.5 * height:
                    line.append(it)
                    continue
            lines.append([it])
        return [it[3] for line in lines for it in sorted(line, key=lambda i: i[2])]

    def _sanitize_candidates(self, candidates):
        """Restrict OCR output to the characters the print can contain.

        - A candidate with no allowed (non-space) characters at all is
          dropped: it's a noise token (typically a lone CJK character the
          multilingual recognizer produced from a speck or embossing).
        - A candidate that is mostly valid but contains disallowed
          characters keeps its length, with each disallowed character
          replaced by "?". Deliberately NOT simply deleted: deleting could
          turn a smudged/misprinted code into a string that happens to
          match the expected value on the PLC (a false accept). A "?"
          guarantees an exact-match comparison fails, so the product is
          rejected - the safe direction for an inspection system.
        """
        allowed = self._allowed_chars
        if not allowed:
            return candidates
        cleaned = []
        for cand in candidates:
            text = unicodedata.normalize("NFKC", str(cand.get("text", "")))
            if not any(ch in allowed and not ch.isspace() for ch in text):
                log.debug("OCR output filter: dropped token %r (no allowed characters)", text)
                continue
            filtered = "".join(ch if ch in allowed else "?" for ch in text).strip()
            if filtered != text:
                log.debug("OCR output filter: %r -> %r", text, filtered)
            cleaned.append({**cand, "text": filtered})
        return cleaned

    @staticmethod
    def _score_candidates(candidates):
        """Aggregate score used to compare two candidate sets (a crop vs
        its 180-degree-rotated twin, see _recognize()'s orientation retry).

        Confidence-weighted character count, not a plain confidence sum.
        The plain sum had a specific, observed failure: on a blurry crop,
        the WRONG (upside-down) orientation collapsed to a single short
        garbage token at moderate confidence ('307312156' @ 0.665, 9
        chars), while the correct orientation split into two lower-scored
        but much longer lines. Summing bare confidences let the short
        garbage win - confirmed in a real capture where
        used_flipped_orientation=True and the saved winning crop was
        visibly upside-down, i.e. the retry took a correctly-oriented crop
        and actively flipped it the wrong way.

        Weighting each candidate's confidence by how many characters it
        actually recovered fixes that: a longer, moderately-confident read
        beats a short, confidently-garbled one. When both orientations
        recover the same amount of text (the common case - e.g. a crop
        that's already upright, where both readings are full-length) the
        character counts are equal, so this reduces to the same ordering
        as the old confidence sum and changes nothing.
        """
        return sum(c["confidence"] * len(c["text"].strip()) for c in candidates)

    def _prepare(self, frame, dynamic_roi=False):
        """ROI/locate + deskew + downscale, done once per frame even when
        several preprocessing variants are read. Returns
        (raw_crop, cycle_debug_dir, profile_data)."""
        profile_data = {}
        self._last_located = None
        t0 = time.perf_counter()
        roi_frame, cycle_debug_dir = self._apply_roi(frame, dynamic=dynamic_roi)
        located = self._last_located if dynamic_roi else None
        profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        roi_frame = self._downscale_roi(roi_frame)
        profile_data["downscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        # Save the crop as it stands BEFORE any preprocessing (raw pixels,
        # for tuning classical-CV stages offline against real frames;
        # ocr_result*.png shows what the recognizer actually got).
        if cycle_debug_dir:
            try:
                os.makedirs(cycle_debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(cycle_debug_dir, "crop_raw_preprocess.png"), roi_frame)
            except Exception as exc:  # pragma: no cover - debug aid only
                log.warning("Raw crop debug write failed (non-fatal): %s", exc)
        return roi_frame, cycle_debug_dir, profile_data, located

    def _frame_variant_input(self, frame, located, variant):
        """Raw input for a FRAME_VARIANTS read (see there)."""
        if variant == "ocr_pipeline":
            # Reuse the OCR pipeline's own static-ROI crop, unmodified. A copy,
            # because that path may draw its debug ROI box onto the frame.
            crop, _ = self._apply_roi(frame.copy(), dynamic=False)
            return self._downscale_roi(crop)
        if variant == "full_frame" or not located:
            return self._downscale_roi(frame)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in located["bbox"])
        margin = int(max(40, self.region_margin_frac * max(x2 - x1, y2 - y1)))
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
        return self._downscale_roi(frame[y1:y2, x1:x2])

    def _preprocess_variant(self, crop, variant, dynamic_roi, profile_data):
        if variant is None:
            if dynamic_roi and self.dot_text_enabled:
                variant = "dot"
            elif dynamic_roi and self.derib_enabled:
                variant = "classic_derib"
            else:
                variant = "classic"
        if variant in ("region", "full_frame", "ocr_pipeline"):
            return self._preprocess_image(crop)  # OCR-pipeline preprocessing, unchanged
        if variant == "region_dot":
            return clean_dot_text(crop, dict(self.dot_text_cfg or {}, illumination="divide"))
        if variant.startswith("dot"):
            cfg = dict(self.dot_text_cfg or {})
            if variant.startswith("dot_divide"):
                cfg["illumination"] = "divide"
            elif variant.startswith("dot_subtract"):
                cfg["illumination"] = "subtract"
            if variant.endswith("_x3") and max(crop.shape[:2]) <= 400:
                cfg["upscale"] = 3  # only on a real text crop, not on a full-frame fallback
            if variant.endswith("_bridge"):
                cfg["bridge_sigma_px"] = 0.7
            if "_tilt_" in variant:
                angle = 3.0 if variant.endswith("_p3") else -3.0
                h, w = crop.shape[:2]
                rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                crop = cv2.warpAffine(crop, rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return clean_dot_text(crop, cfg)
        if variant == "classic_derib":
            t_derib = time.perf_counter()
            crop = self._derib_crop(crop)
            profile_data["derib_ms"] = round((time.perf_counter() - t_derib) * 1000, 3)
        return self._preprocess_image(crop)

    def _recognize(self, frame, profile=False, dynamic_roi=False, variant=None, prepared=None):
        """
        Shared pipeline: ROI, downscale, and the actual PaddleOCR inference call.

        Returns (candidates, elapsed_ms, profile_data, engine_error):
        every text candidate PaddleOCR produced (no keyword filtering, but
        restricted to ocr.allowed_chars - see _sanitize_candidates), timing
        info, and engine_error - None normally, or a message when the OCR
        engine itself failed, in which case candidates is empty and the
        caller must report a technical failure rather than "no text".

        Both `run()` (keyword match, for the OCR mode) and
        `read()` (raw read-out, for the OCRead mode) build on this so the
        downscale/preprocessing/inference path is identical between the two
        modes — only the ROI step differs, via dynamic_roi (OCRead only).

        For dynamic_roi (OCRead) specifically, two extra passes can run,
        each targeting a documented failure mode rather than general
        robustness padding - see _derib_crop() and the orientation-retry
        block below for the evidence behind each:
          - rib inpainting on the crop itself (derib_enabled)
          - a low-confidence retry against a 180-degree-rotated copy of
            the crop (orientation_retry_enabled), since deskew_crop() only
            resolves line tilt, not whether the line came out upside-down.
        """
        # Collected unconditionally now (a handful of perf_counter() calls
        # costs nothing measurable) rather than only when profile=True, so
        # the slow-cycle logging below always has a real stage breakdown to
        # show - see self.slow_cycle_log_threshold_ms in __init__. profile=
        # still controls whether this comes back in the returned result
        # dict for benchmarks/ocr_benchmark.py.
        # Timings are collected unconditionally (cheap) so the slow-cycle log
        # always has a stage breakdown; profile= only controls whether they
        # are returned in the result (benchmarks/ocr_benchmark.py).
        if prepared is None:
            prepared = self._prepare(frame, dynamic_roi)
        raw_crop, cycle_debug_dir, prep_profile, located = prepared
        profile_data = dict(prep_profile)

        t1b = time.perf_counter()
        if variant in self.FRAME_VARIANTS:
            raw_crop = self._frame_variant_input(frame, located, variant)
        roi_frame = self._preprocess_variant(raw_crop, variant, dynamic_roi, profile_data)
        profile_data["preprocess_ms"] = round((time.perf_counter() - t1b) * 1000, 3)

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            log.debug("PaddleOCR input: shape=%s dtype=%s", roi_frame.shape, roi_frame.dtype)

        start_time = time.perf_counter()

        t2 = start_time
        # engine_error is set when the OCR engine itself failed (worker
        # down, hung, crashed, raised). It is returned to run()/read(),
        # which put it in the result's "error" field so vision.py reports
        # NOK with failure_reason="ocr_engine_error". Previously this was
        # swallowed into an empty candidate list, which made a dead engine
        # indistinguishable from a bottle with no print on it.
        engine_error = None
        try:
            # paddle_worker.py parses PaddleOCR's own result object inside
            # the worker process now (see its module docstring) and hands
            # back a plain list of {"text", "confidence"} dicts - nothing
            # further to unpack here regardless of which PaddleOCR version
            # produced it.
            candidates = self._order_candidates(self._sanitize_candidates(self._paddle.ocr(roi_frame) or []))
            if debug_enabled:
                log.debug("PaddleOCR result count=%s", len(candidates))
        except PaddleWorkerError as e:
            log.error("PaddleOCR engine failure: %s", e)
            candidates = []
            engine_error = str(e)
        except Exception as e:
            log.error("PaddleOCR.ocr() failed: %s", e, exc_info=True)
            candidates = []
            engine_error = f"{type(e).__name__}: {e}"

        profile_data["ocr_ms"] = round((time.perf_counter() - t2) * 1000, 3)

        # Orientation retry - see the docstring above and the comment
        # above self.orientation_retry_enabled in __init__. Only pays for
        # a second inference pass when the first one looks doubtful
        # (nothing found, or any candidate below threshold); a clean
        # upright read never takes this branch.
        used_flipped = False
        retry_audit = None
        # Never retry after an engine failure: the flipped pass would hit the
        # same broken engine and just double the time this cycle takes.
        if dynamic_roi and self.orientation_retry_enabled and engine_error is None:
            should_retry = not candidates
            if self.orientation_retry_require_low_confidence and candidates:
                worst_conf = min(c["confidence"] for c in candidates)
                should_retry = should_retry or worst_conf < self.orientation_retry_confidence_threshold
            if should_retry:
                t_retry = time.perf_counter()
                flipped_candidates = []
                flipped_frame = None
                try:
                    flipped_frame = cv2.rotate(roi_frame, cv2.ROTATE_180)
                    flipped_candidates = self._order_candidates(
                        self._sanitize_candidates(self._paddle.ocr(flipped_frame) or []))
                except Exception as exc:
                    flipped_frame = None
                    if not candidates:
                        # Nothing usable from the first pass and the engine
                        # failed on the second: we can't claim "no text".
                        log.error("Orientation retry failed and first pass was empty: %s", exc)
                        engine_error = f"orientation retry failed: {exc}"
                    else:
                        log.warning("Orientation retry failed (non-fatal, keeping first pass): %s", exc)
                profile_data["orientation_retry_ms"] = round((time.perf_counter() - t_retry) * 1000, 3)

                # Record BOTH sides of the comparison, not just the winner.
                # Without this a debug capture only shows the chosen result,
                # so a wrong choice can't be told apart from a case where
                # both orientations read badly - which is exactly the
                # ambiguity that made the previous scoring bug take several
                # sessions to pin down.
                orig_score = self._score_candidates(candidates)
                flip_score = self._score_candidates(flipped_candidates)
                retry_audit = (
                    f"retry_original_score={orig_score:.2f} "
                    f"original_text={' '.join(c['text'] for c in candidates).strip()!r} | "
                    f"retry_flipped_score={flip_score:.2f} "
                    f"flipped_text={' '.join(c['text'] for c in flipped_candidates).strip()!r}"
                )
                log.debug("Orientation retry: %s", retry_audit)

                if flipped_frame is not None and flip_score > orig_score:
                    candidates = flipped_candidates
                    roi_frame = flipped_frame  # keep debug capture consistent with what was actually read
                    used_flipped = True
                    log.debug("Orientation retry: flipped crop scored higher, using it.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        profile_data["total_ocr_call_ms"] = round(elapsed_ms, 3)

        # Auto-log the stage breakdown for any cycle that crossed the slow
        # threshold, independent of whether the caller passed profile=True -
        # this is what a spike capture was missing so far: elapsed_ms alone
        # doesn't say whether the time went into Paddle's own compute
        # (ocr_ms) or somewhere it shouldn't have (a large gap between
        # total_ocr_call_ms and ocr_ms+orientation_retry_ms points at
        # queueing/IPC/system contention rather than model compute).
        if elapsed_ms > self.slow_cycle_log_threshold_ms:
            log.warning(
                "PaddleOCR slow cycle: elapsed_ms=%.1f (threshold=%.0f) stage_breakdown=%s",
                elapsed_ms, self.slow_cycle_log_threshold_ms, profile_data,
            )

        if cycle_debug_dir:
            self._write_ocr_result_debug(
                cycle_debug_dir, roi_frame, candidates, elapsed_ms,
                name=f"ocr_result_{variant}.png" if variant else "ocr_result.png",
                extra_lines=[
                    f"used_flipped_orientation={used_flipped}",
                ] + ([retry_audit] if retry_audit else []),
            )

        return candidates, elapsed_ms, profile_data, engine_error

    def run(self, frame, profile=False):
        """
        Executes OCR on the provided frame, applying ROI and preprocessing if configured.

        Used by the OCR mode: recognized text is matched against the
        configured/active keyword (and date pattern) here, and only matching
        candidates are returned as detections.
        """
        candidates, elapsed_ms, profile_data, engine_error = self._recognize(frame, profile=profile)

        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )
        keyword_set = {k.lower() for k in keywords if isinstance(k, str)} if self.app_state else self.keyword_set

        t3 = time.perf_counter()
        detections = []
        for candidate in candidates:
            text = candidate["text"]
            word = text.lower()
            if keyword_set and word not in keyword_set:
                if self._date_pattern and not self._date_pattern.search(text):
                    continue
            detections.append(candidate)

        if profile:
            profile_data["filter_ms"] = round((time.perf_counter() - t3) * 1000, 3)
            profile_data["total_ms"] = round(sum(v for v in profile_data.values() if isinstance(v, (int, float))), 3)

        processing_time_ms = round(elapsed_ms, 1)
        searched_word = keywords[0] if keywords else ""
        log.debug(
            "PaddleOCR run completed: processing_time_ms=%s candidates=%s detections=%s searched_word=%r",
            processing_time_ms,
            len(candidates),
            len(detections),
            searched_word,
        )
        result = {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": searched_word
        }
        if engine_error:
            result["error"] = engine_error

        if profile:
            result["_profile_ms"] = profile_data

        return result

    def _read_with_font_reader(self, frame, profile=False):
        """OCRead with the trained dot-font reader (see font_reader.py)."""
        t0 = time.perf_counter()
        crop, cycle_debug_dir, profile_data, located = self._prepare(frame, dynamic_roi=True)
        r = self.font_reader.read(crop, self.validator.lines)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if r is None:
            reason, candidates, verdict = REJECT_FORMAT, [], None
        else:
            candidates = [{"text": line, "confidence": r["confidence"]} for line in r["lines"]]
            # The font reader has its own gates (literal_score, digit confidence);
            # the template check here adds the value_pattern sanity check.
            verdict = self.validator.validate([{"text": line, "confidence": 1.0} for line in r["lines"]])
            if r["literal_score"] < self.font_reader.min_literal_score:
                reason = REJECT_FORMAT
            elif r["confidence"] < self.font_reader.min_digit_confidence:
                reason = REJECT_CONFIDENCE
            else:
                reason = verdict["reason"]
        accepted = reason is None
        if not accepted and located is None and reason == REJECT_FORMAT:
            reason = "no_text_detected"   # no print found on the frame and nothing readable
        # A rejected read must not show a code: the reader always fills every
        # digit position with its best guess, so the decode of a rejected
        # read is not a reading (it looked like guessing on the HMI). It is
        # kept only in the debug file.
        text = r["text"] if accepted else ""
        if not accepted:
            candidates = []
        read_summary = {"variant": "font_reader", "accepted": accepted, "reason": reason,
                        "text": r["text"] if accepted else None, "raw_text": text,
                        "min_token_confidence": r["confidence"] if r else None,
                        "literal_score": r["literal_score"] if r else None,
                        "timings_ms": r.get("timings_ms") if r else None,
                        "orientation": ("reversed" if r and r["flipped"] else "normal") if r else None}
        if cycle_debug_dir:
            try:
                os.makedirs(cycle_debug_dir, exist_ok=True)
                with open(os.path.join(cycle_debug_dir, "font_reader.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"{read_summary}\n")
                    if r is not None and not accepted:
                        fh.write(f"rejected decode (not a reading): {r['text']!r}\n")
            except Exception as exc:  # pragma: no cover - debug aid only
                log.warning("Font reader debug write failed (non-fatal): %s", exc)
        result = {
            "detections": candidates,
            "text": text,
            "processing_time_ms": round(elapsed, 1),
            "mode": "ocread",
            "validation": {"accepted": accepted, "reason": reason, "reads": [read_summary], "fusion": None},
        }
        if not accepted:
            result["failure_reason"] = reason
        if profile:
            result["_profile_ms"] = profile_data
        return result

    def read(self, frame, profile=False):
        """
        OCRead: read the printed code from the frame.

        Without ocr.read_validation: one read, every recognized token joined
        as-is (any text = OK; the PLC does the comparison).

        With ocr.read_validation (production): the crop is prepared once and
        read with up to N preprocessing variants (consensus.variants). Each
        read is fielded against the print template (code_validator.py). The
        result is OK only when at least consensus.min_agree reads produce
        the same validated code; "text" is then the canonical code (e.g.
        "4113 08:26 BB 05/26"). Otherwise status NOK with failure_reason
        format_mismatch / low_confidence / invalid_value / no_consensus /
        no_text_detected, and "text" holds the raw read for display only.
        Stops as soon as the outcome is decided, so a clean bottle costs
        min_agree reads, not N.
        """
        if self.validator is None:
            candidates, elapsed_ms, profile_data, engine_error = self._recognize(frame, profile=profile, dynamic_roi=True)
            full_text = " ".join(c["text"] for c in candidates).strip()
            log.debug("PaddleOCR read completed: processing_time_ms=%.1f detections=%s text=%r",
                      elapsed_ms, len(candidates), full_text)
            result = {
                "detections": candidates,
                "text": full_text,
                "processing_time_ms": round(elapsed_ms, 1),
                "mode": "ocread",
            }
            if engine_error:
                result["error"] = engine_error
            if profile:
                result["_profile_ms"] = profile_data
            return result

        if self.font_reader is not None:
            return self._read_with_font_reader(frame, profile)

        prepared = self._prepare(frame, dynamic_roi=True)
        variants = self.consensus_variants
        need = self.consensus_min_agree
        reads, votes = [], Counter()
        total_ms, profile_data = 0.0, {}
        for i, variant in enumerate(variants):
            candidates, elapsed_ms, profile_data, engine_error = self._recognize(
                frame, profile=profile, dynamic_roi=True, variant=variant, prepared=prepared)
            total_ms += elapsed_ms
            if engine_error:
                result = {"detections": candidates, "text": "", "processing_time_ms": round(total_ms, 1),
                          "mode": "ocread", "error": engine_error}
                if profile:
                    result["_profile_ms"] = profile_data
                return result
            verdict = self.validator.validate(candidates)
            reads.append({"variant": variant, "candidates": candidates, **verdict})
            if self.consensus_fusion == "position_vote" or not self.consensus_early_stop:
                continue  # accuracy first: read every variant, decide below
            if verdict["accepted"]:
                votes[verdict["text"]] += 1
                if votes[verdict["text"]] >= need:
                    break
            remaining = len(variants) - i - 1
            if max(votes.values(), default=0) + remaining < need:
                break

        primary = reads[0]
        fusion_info = None
        if self.consensus_fusion == "position_vote":
            fused = self.validator.fuse(reads, min_reads=self.consensus_min_reads,
                                        min_share=self.consensus_min_share)
            fusion_info = {k: fused[k] for k in ("n_reads", "min_share", "position_shares") if k in fused}
            accepted = fused["accepted"]
            if accepted:
                text, reason = fused["text"], None
                source = max((r for r in reads if r.get("text") == fused["text"]),
                             key=lambda r: r.get("min_token_confidence") or 0.0, default=primary)
            else:
                source = primary
                reason = fused["reason"]
                text = fused.get("text") or primary["raw_text"]
        else:
            if not self.consensus_early_stop:
                votes = Counter(r["text"] for r in reads if r["accepted"])
            ranked = votes.most_common(2)
            winner, count = ranked[0] if ranked else (None, 0)
            # A tie between two different codes is a disagreement, not a pass.
            tied = len(ranked) > 1 and ranked[1][1] == count
            accepted = winner is not None and count >= need and not tied
            if accepted:
                source = next(r for r in reads if r["accepted"] and r.get("text") == winner)
                text, reason = winner, None
            else:
                source = primary
                text = primary["raw_text"]
                reason = REJECT_CONSENSUS if votes else primary["reason"]

        summary = [
            {k: r.get(k) for k in ("variant", "accepted", "reason", "text", "raw_text", "corrections",
                                   "literal_fixes", "ignored_tokens", "orientation", "min_token_confidence")}
            for r in reads
        ]
        log.debug("OCRead validation: accepted=%s text=%r reason=%s reads=%s", accepted, text, reason, summary)
        result = {
            "detections": source["candidates"],
            "text": text,
            "processing_time_ms": round(total_ms, 1),
            "mode": "ocread",
            "validation": {"accepted": accepted, "reason": reason, "reads": summary, "fusion": fusion_info},
        }
        if not accepted:
            result["failure_reason"] = reason
        if profile:
            result["_profile_ms"] = profile_data
        return result
