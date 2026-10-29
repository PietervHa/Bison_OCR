"""
OCRead engine

Reads the dot-printed code on a bottle cap:

    locate the print (text_locator.py) -> deskewed crop -> font reader
    (font_reader.py, which cleans the crop with dot_text.py), once per
    configured print format -> pick one format -> format check
    (code_validator.py: template + value_pattern) -> result

Several print formats can be configured (ocr.read_validation.templates),
e.g. "#### ##:##" / "BB ##/##", "#### ##:##" / "EXP ##/##" and the single
line "#### ##:##". Every format the model supports is read on the same
crop, with the work they share (ink map, deskew, line bands, line 1's grid
and cells) done once per frame (font_reader.PreparedCrop). Of the trusted
reads (the font reader's own gates pass), the format with more lines wins,
then the higher literal score: a two-line print is never accepted as the
single-line format with line 2 ignored. The chosen format is in
validation["format"]; validation["reads"][0] has the time per step
("step_ms") and per format tried ("formats_tried"[i]["ms"]).

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
from backend.detection.ocr.code_validator import (CodeTemplate, REJECT_FORMAT, REJECT_CONFIDENCE, REJECT_NO_TEXT,
                                                  REJECT_TIMEOUT)
from backend.detection.ocr.font_reader import FontReader, PreparedCrop, ReadTimeout
from backend.detection.ocr.text_locator import locate_text_region, deskew_crop, row_profile_rect
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
        # it fits one of the configured print formats; otherwise NOK with a
        # reason instead of forwarding a guess to the PLC.
        val_cfg = ocr_cfg.get("read_validation") if isinstance(ocr_cfg.get("read_validation"), dict) else {}
        self.validators = CodeTemplate.list_from_config(val_cfg)
        # Safety net: a read still running after this many ms (from the
        # start of read(), incl. locating the print) stops and is NOK
        # "timeout" - never a code. 0 = no limit.
        self.time_budget_ms = float(val_cfg.get("time_budget_ms", 1500) or 0)
        self.skipped_formats = []
        self._last_located = None
        # Second attempt at the print's row angle for a read that failed
        # (see read()); accepted only past these stricter gates.
        self.row_angle_retry = bool(dynamic_roi_cfg.get("row_angle_retry", False))
        self.row_angle_retry_min_diff_deg = float(dynamic_roi_cfg.get("row_angle_retry_min_diff_deg", 3.0))
        self.row_angle_retry_min_confidence = float(dynamic_roi_cfg.get("row_angle_retry_min_digit_confidence", 0.9))
        self.row_angle_retry_min_literal = float(dynamic_roi_cfg.get("row_angle_retry_min_literal_score", -0.05))

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
            # A format with characters the model doesn't know (e.g. EXP with
            # a model trained without E, X, P) is skipped, not an error.
            self.skipped_formats = [v.lines for v in self.validators
                                    if not font_reader.supports(v.lines) or len(v.lines) not in (1, 2)]
            if len(self.skipped_formats) == len(self.validators):
                self.unavailable_reason = ("font reader does not support any template "
                                           f"{[v.lines for v in self.validators]}")
            else:
                self.font_reader = font_reader
        except Exception as exc:
            self.unavailable_reason = f"font reader could not be loaded: {exc}"

        if self.font_reader is None:
            log.error("OCRead unavailable (%s) - every read will be NOK (ocr_engine_error).",
                      self.unavailable_reason)
        else:
            log.info("OCRead: font reader, formats=%s", [f.lines for f in self._read_formats()])
            for lines in self.skipped_formats:
                log.warning("OCRead: format %s skipped - the font reader model does not support it", lines)

    @property
    def validator(self):
        """The first configured format. Assigning one replaces all formats."""
        return self.validators[0]

    @validator.setter
    def validator(self, value):
        self.validators = [value]

    def _read_formats(self):
        return [v for v in self.validators if v.lines not in self.skipped_formats]

    def _read_all_formats(self, crop, deadline=None):
        """Read the crop once per supported format and pick one.

        Returns (validator, read, attempts, step_ms): the trusted read of the
        format with the most lines (then the highest literal score); with no
        trusted read, the read with the highest literal score (for its reject
        reason); (None, None, attempts, step_ms) when nothing could be read at
        all. step_ms: time per font reader step over all formats (ink, deskew,
        band_fit, cell_fit, cells); attempts carry each format's own time.

        All formats share one PreparedCrop: what they have in common is
        computed by the first format that needs it, the others reuse it.
        deadline (time.perf_counter()): past it, ReadTimeout is raised, with
        the attempts and step_ms so far as its .attempts / .step_ms."""
        reads, attempts = [], []
        prep = PreparedCrop(crop, deadline)
        # More lines first: once a format is trusted, formats with fewer lines
        # can no longer win and are not read (saves a read per cycle).
        for v in sorted(self._read_formats(), key=lambda v: -len(v.lines)):
            if any(t and len(x.lines) > len(v.lines) for x, _, t in reads):
                break
            ts = time.perf_counter()
            try:
                r = self.font_reader.read(prep, v.lines)
            except ReadTimeout as exc:
                attempts.append({"format": v.lines, "trusted": False, "literal_score": None,
                                 "min_token_confidence": None, "ms": round((time.perf_counter() - ts) * 1000, 1),
                                 "timeout": True})
                exc.attempts, exc.step_ms = attempts, prep.timings_ms
                raise
            ms = round((time.perf_counter() - ts) * 1000, 1)
            if r is not None and len(r["lines"]) != len(v.lines):
                r = None
            trusted = (r is not None and r["literal_score"] >= self.font_reader.min_literal_score
                       and r["confidence"] >= self.font_reader.min_digit_confidence)
            attempts.append({"format": v.lines, "trusted": trusted,
                             "literal_score": r["literal_score"] if r else None,
                             "min_token_confidence": r["confidence"] if r else None, "ms": ms})
            if r is not None:
                reads.append((v, r, trusted))
        if not reads:
            return None, None, attempts, prep.timings_ms
        trusted = [x for x in reads if x[2]]
        if trusted:
            v, r, _ = max(trusted, key=lambda x: (len(x[0].lines), x[1]["literal_score"]))
        else:
            v, r, _ = max(reads, key=lambda x: x[1]["literal_score"])
        return v, r, attempts, prep.timings_ms

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

    def _check(self, validator, r):
        """Reject reason for the chosen read of _read_all_formats: the font
        reader's own gates, then the template check. None = accepted."""
        if r is None:
            return REJECT_FORMAT
        # The font reader has its own gates (literal_score, digit confidence);
        # the template check here adds the value_pattern sanity check.
        verdict = validator.validate([{"text": line, "confidence": 1.0} for line in r["lines"]])
        if r["literal_score"] < self.font_reader.min_literal_score:
            return REJECT_FORMAT
        if r["confidence"] < self.font_reader.min_digit_confidence:
            return REJECT_CONFIDENCE
        return verdict["reason"]

    def _row_angle_retry(self, frame, located, deadline):
        """Second attempt for a read that failed: deskew at the angle of the
        print's rows (text_locator.row_profile_rect) instead of the
        minAreaRect angle, and read all formats again within the same
        deadline. The minAreaRect box was tilted 10-50 degrees off a level
        print on caps with a rib or embossing in the ink (33 of the 514 test
        images); the first attempt is never replaced by this, so a read it
        accepts stays exactly as it was.

        Returns (validator, read, attempts, step_ms, angle) when the retry
        passes every check of a normal read AND the stricter retry gates
        (row_angle_retry_min_digit_confidence / _min_literal_score: a
        misfitted cell grid can still fit the template with one digit
        guessed, e.g. "EXP 06/28" for "EXP 05/28" at digit confidence 0.16,
        where accepted first reads are >= 0.95 in 99% of the test images);
        otherwise None. Also None when the row angle is within
        row_angle_retry_min_diff_deg of the first attempt's (same crop)."""
        points = located.get("ink_points")
        if points is None or len(points) < 3:
            return None
        padding = int(self.dynamic_roi_cfg.get("padding_px", 10))
        rect = row_profile_rect(points, padding)
        first = located["rotated_rect"]["angle"]
        if abs((rect["angle"] - first + 90.0) % 180.0 - 90.0) < self.row_angle_retry_min_diff_deg:
            return None
        crop = deskew_crop(frame, rect, extra_padding_px=20)
        if crop is None:
            return None
        try:
            validator, r, attempts, step_ms = self._read_all_formats(self._downscale_roi(crop), deadline)
        except ReadTimeout:
            return None
        reason = self._check(validator, r)
        if (reason is not None or r["confidence"] < self.row_angle_retry_min_confidence
                or r["literal_score"] < self.row_angle_retry_min_literal):
            log.debug("Row angle retry (%.1f deg) not accepted: %s", rect["angle"],
                      reason or f"literal {r['literal_score']} confidence {r['confidence']}")
            return None
        return validator, r, attempts, step_ms, rect["angle"]

    def read(self, frame, profile=False):
        """
        Read the printed code from the frame.

        OK only when the font reader's own gates (literal_score, digit
        confidence) and the template check (format + value_pattern) all
        pass for the chosen format; "text" is then the canonical code, e.g.
        "4113 08:26 BB 05/26" or "6286 16:24".
        Otherwise failure_reason is format_mismatch / low_confidence /
        invalid_value / no_text_detected / timeout and "text" is empty.

        no_text_detected straight away when the locator (dynamic ROI) found
        no print: reading the full frame instead took 1.2-1.7 s and never
        gave a code (22 of the 514 test images). timeout: the read was still
        running after ocr.read_validation.time_budget_ms.
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
        validator, r, attempts, step_ms, reason = None, None, [], {}, None
        retry_angle = None
        if located is None and self.dynamic_roi_enabled:
            reason = REJECT_NO_TEXT            # no print found: nothing to read
        else:
            deadline = t0 + self.time_budget_ms / 1000.0 if self.time_budget_ms > 0 else None
            try:
                validator, r, attempts, step_ms = self._read_all_formats(crop, deadline)
            except ReadTimeout as exc:
                attempts, step_ms, reason = exc.attempts, exc.step_ms, REJECT_TIMEOUT
            if reason is None:
                reason = self._check(validator, r)
                if reason is not None and self.row_angle_retry and located is not None:
                    retry = self._row_angle_retry(frame, located, deadline)
                    if retry is not None:
                        validator, r, retry_attempts, step_ms, retry_angle = retry
                        attempts = attempts + [dict(a, retry=True) for a in retry_attempts]
                        reason = None
        elapsed = (time.perf_counter() - t0) * 1000.0
        accepted = reason is None
        if not accepted and located is None and reason == REJECT_FORMAT:
            reason = "no_text_detected"   # no print found on the frame and nothing readable
        # A rejected read must not show a code: the reader always fills every
        # digit position with its best guess, so the decode of a rejected
        # read is not a reading (it looked like guessing on the HMI). It is
        # kept only in the debug file.
        text = r["text"] if accepted else ""
        candidates = [{"text": line, "confidence": r["confidence"]} for line in r["lines"]] if accepted else []
        read_summary = {"variant": "font_reader", "accepted": accepted, "reason": reason,
                        "text": r["text"] if accepted else None, "raw_text": text,
                        "min_token_confidence": r["confidence"] if r else None,
                        "literal_score": r["literal_score"] if r else None,
                        "timings_ms": r.get("timings_ms") if r else None,
                        # Per step over all formats, incl. locating the print.
                        "step_ms": {"locate": round(profile_data.get("roi_ms", 0.0), 1), **step_ms},
                        "orientation": ("reversed" if r and r["flipped"] else "normal") if r else None,
                        "format": validator.lines if validator else None, "formats_tried": attempts,
                        "row_angle_retry": retry_angle}
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
            "validation": {"accepted": accepted, "reason": reason, "reads": [read_summary], "fusion": None,
                           "format": validator.lines if validator else None},
        }
        if not accepted:
            result["failure_reason"] = reason
        if profile:
            result["_profile_ms"] = profile_data
        return result
