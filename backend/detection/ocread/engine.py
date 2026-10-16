"""
OCRead engine

Reads the dot-printed code on a bottle cap:

    locate the print (text_locator.py) -> deskewed crop -> font reader
    (font_reader.py, which cleans the crop with dot_text.py) -> format
    check (code_validator.py: template + value_pattern) -> result

The result is a dict with the keys downstream code relies on: detections,
text, processing_time_ms, mode="ocread", validation and - for a rejected
read - failure_reason. backend/core/vision.py turns it into OK/NOK.

A rejected read never carries a code: the font reader always fills every
digit position with its best guess, so the decode of a rejected read is
not a reading. It is only written to the debug capture.
"""

import os
import time
from pathlib import Path

import cv2

from backend.core.config_loader import cfg
from backend.detection.ocr.code_validator import CodeTemplate, REJECT_FORMAT, REJECT_CONFIDENCE
from backend.detection.ocr.font_reader import FontReader
from backend.detection.ocr.text_locator import locate_text_region, deskew_crop
from backend.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODE = "ocread"


def resolve_project_path(path) -> str:
    """Relative paths in the config are relative to the project root - the
    same as data/results and data/logs - NOT to whatever folder the app was
    started from. (debug_dir used to be taken relative to the working
    directory, so starting the app from backend/ silently wrote the debug
    captures to backend/data/debug/dynamic_roi.)"""
    p = Path(str(path))
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


class OCRead:
    """
    Reads the printed code from a camera frame with the trained dot-font reader.
    """
    def __init__(self):
        ocr_cfg = cfg["ocr"]
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])

        # Dynamic ROI: locates the printed text block via classical CV,
        # since the text doesn't land in the same place on every bottle.
        # See text_locator.py. When disabled, or when nothing is found on a
        # given frame, the full frame is read.
        dynamic_roi_cfg = ocr_cfg.get("dynamic_roi", {}) if isinstance(ocr_cfg.get("dynamic_roi"), dict) else {}
        self.dynamic_roi_enabled = bool(dynamic_roi_cfg.get("enabled", False))
        self.dynamic_roi_cfg = dynamic_roi_cfg
        self.dynamic_roi_debug_dir = (
            resolve_project_path(dynamic_roi_cfg.get("debug_dir") or "data/debug/dynamic_roi")
            if dynamic_roi_cfg.get("debug_save") else None
        )
        if self.dynamic_roi_debug_dir:
            log.info("OCRead debug captures are written to %s", self.dynamic_roi_debug_dir)

        # Read validation (code_validator.py): a read is only reported OK if
        # it fits the configured print template; otherwise NOK with a reason
        # instead of forwarding a guess to the PLC.
        val_cfg = ocr_cfg.get("read_validation") if isinstance(ocr_cfg.get("read_validation"), dict) else {}
        self.validator = CodeTemplate.from_config(val_cfg)
        self._last_located = None

        # Font reader: classifier trained on this printer's dot font
        # (font_reader.py). If it can't be used, every read is reported as a
        # technical failure (NOK, failure_reason "ocr_engine_error") rather
        # than a guess.
        self.font_reader = None
        self.unavailable_reason = None
        fr_cfg = val_cfg.get("font_reader") if isinstance(val_cfg.get("font_reader"), dict) else {}
        try:
            model = fr_cfg.get("model") or None
            font_reader = FontReader(
                resolve_project_path(model) if model else None,
                min_literal_score=float(fr_cfg.get("min_literal_score", -1.0)),
                min_digit_confidence=float(fr_cfg.get("min_digit_confidence", 0.0)),
            )
            if not font_reader.supports(self.validator.lines) or len(self.validator.lines) != 2:
                self.unavailable_reason = f"font reader does not support template {self.validator.lines}"
            else:
                self.font_reader = font_reader
        except Exception as exc:
            self.unavailable_reason = f"font reader could not be loaded: {exc}"

        if self.font_reader is None:
            log.error("OCRead unavailable (%s) - every read will be NOK (ocr_engine_error).",
                      self.unavailable_reason)
        else:
            log.info("OCRead: font reader, template=%s", self.validator.lines)

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

    def _apply_roi(self, frame):
        """Crop the frame to the text region.

        It only ever does one of two things: crop to wherever
        locate_text_region() actually found ink on this frame (deskewed via
        deskew_crop), or - if it found nothing at all - hand the full,
        uncropped frame to the reader rather than guessing a fixed
        rectangle. A production line where bottles don't land the same way
        every time means a static box is wrong as often as it's right, so
        there is no static-ROI fallback at all. No debug overlay is drawn
        here either - the detected region changes on every frame, so a box
        drawn on the live view wouldn't mean anything to line staff.

        Returns (crop, cycle_debug_dir). cycle_debug_dir is the same
        per-cycle folder text_locator.py just wrote its own debug images
        into (None if dynamic_roi debug_save is off) - read() writes the
        read result alongside the localization debug images, so a debug
        capture shows both where the crop came from and what was read.
        """
        if not self.dynamic_roi_enabled:
            log.warning(
                "OCRead called with dynamic ROI disabled (ocr.dynamic_roi.enabled=false) - "
                "running OCR on the full frame. Enable it in config to locate text automatically."
            )
            self._last_located = None
            return frame, None

        located = locate_text_region(frame, self.dynamic_roi_cfg, debug_dir=self.dynamic_roi_debug_dir)
        self._last_located = located or None
        if not located:
            log.debug("Dynamic ROI: no text region found on this frame - running OCR on the full frame.")
            return frame, None

        cycle_debug_dir = located.get("cycle_debug_dir")
        rotated_rect = located.get("rotated_rect")
        # Extra margin here, beyond dynamic_roi.padding_px (which is already
        # baked into rotated_rect). Changing it changes the crop the font
        # reader sees, so re-check reads on real frames before touching it.
        crop = deskew_crop(frame, rotated_rect, extra_padding_px=20) if rotated_rect else None
        if crop is None:
            # Degenerate rotated rect (shouldn't normally happen) - use
            # the axis-aligned box we already computed rather than the
            # full frame, since we did find *something*.
            x1, y1, x2, y2 = located["bbox"]
            crop = frame[y1:y2, x1:x2]

        return crop, cycle_debug_dir

    def _prepare(self, frame):
        """Locate + deskew + downscale. Returns
        (raw_crop, cycle_debug_dir, profile_data, located)."""
        profile_data = {}
        self._last_located = None
        t0 = time.perf_counter()
        roi_frame, cycle_debug_dir = self._apply_roi(frame)
        located = self._last_located
        profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        roi_frame = self._downscale_roi(roi_frame)
        profile_data["downscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        # Save the crop as it stands BEFORE any preprocessing (raw pixels,
        # for tuning classical-CV stages offline against real frames).
        if cycle_debug_dir:
            try:
                os.makedirs(cycle_debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(cycle_debug_dir, "crop_raw_preprocess.png"), roi_frame)
            except Exception as exc:  # pragma: no cover - debug aid only
                log.warning("Raw crop debug write failed (non-fatal): %s", exc)
        return roi_frame, cycle_debug_dir, profile_data, located

    def read(self, frame, profile=False):
        """
        Read the printed code from the frame.

        OK only when the font reader's own gates (literal_score, digit
        confidence) and the template check (format + value_pattern) all
        pass; "text" is then the canonical code, e.g. "4113 08:26 BB 05/26".
        Otherwise failure_reason is format_mismatch / low_confidence /
        invalid_value / no_text_detected and "text" is empty.
        """
        if self.font_reader is None:
            return {
                "detections": [],
                "text": "",
                "processing_time_ms": 0.0,
                "mode": MODE,
                "error": self.unavailable_reason or "font reader unavailable",
            }

        t0 = time.perf_counter()
        crop, cycle_debug_dir, profile_data, located = self._prepare(frame)
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
            "mode": MODE,
            "validation": {"accepted": accepted, "reason": reason, "reads": [read_summary], "fusion": None},
        }
        if not accepted:
            result["failure_reason"] = reason
        if profile:
            result["_profile_ms"] = profile_data
        return result
