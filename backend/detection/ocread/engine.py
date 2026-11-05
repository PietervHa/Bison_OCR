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
import numpy as np

from backend.core.config_loader import cfg
from backend.detection.ocr.code_validator import (CodeTemplate, LINE1, REJECT_FORMAT, REJECT_CONFIDENCE,
                                                  REJECT_IMPLAUSIBLE, REJECT_NO_TEXT, REJECT_TIMEOUT, REJECT_UNSTABLE,
                                                  plausible_line1)
from backend.detection.ocr import grid_jit
from backend.detection.ocr.font_reader import FontReader, PreparedCrop, ReadTimeout
from backend.detection.ocr.ink_channel import InkChannel
from backend.detection.ocr.text_locator import (locate_text_region, locate_text_region_scaled, deskew_crop,
                                                row_profile_rect, alternative_rects, extend_rect_along_line)
from backend.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Agreement check (OCRead._agree): the fixed moves of the read's box per
# extra read, in this order - (dx px, dy px, angle deg, size factor): one of
# each kind of perturbation of the stress test that found reads accepted
# WRONG at digit confidence 0.946-0.972 (shift +/-2 px, turn +/-1 deg,
# size +/-3%), then the opposite turn. With two moves (shift and turn
# only), 3 of 5140 perturbed reads still agreed on a wrong code.
AGREEMENT_MOVES = ((2.0, -2.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.03), (0.0, 0.0, -1.0, 1.0))

MODE = "ocread"


def resolve_project_path(path) -> str:
    """Relative paths in the config are relative to the project root - the
    same as data/results and data/logs - NOT to whatever folder the app was
    started from. (debug_dir used to be taken relative to the working
    directory, so starting the app from backend/ silently wrote the debug
    captures to backend/data/debug/dynamic_roi.)"""
    p = Path(str(path))
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _same_rect(a, b) -> bool:
    """Two rotated rects (text_locator form) that give the same crop:
    centre and size within 2 px, angle within 0.5 degrees."""
    return (max(abs(a["center"][i] - b["center"][i]) for i in (0, 1)) <= 2.0
            and max(abs(a["size"][i] - b["size"][i]) for i in (0, 1)) <= 2.0
            and abs((a["angle"] - b["angle"] + 90.0) % 180.0 - 90.0) <= 0.5)


class OCRead:
    """
    Reads the printed code from a camera frame with the trained dot-font reader.

    config: the whole configuration dict (default: the loaded config file,
    backend.core.config_loader.cfg); a product profile passes the config
    with its overrides merged in (backend/core/profiles.py). profile_name:
    the profile's name, recorded in every result as "profile" (None = not
    recorded).
    """
    def __init__(self, config=None, profile_name=None):
        config = cfg if config is None else config
        ocr_cfg = config["ocr"]
        self.profile_name = profile_name
        # OpenCV's own worker threads (performance.cv_threads): on these
        # small crops 1-8 threads read equally fast, but with more than one
        # every parallel call waits for its slowest worker - under load a
        # busy core stalled the read - and the workers don't run at the
        # vision thread's priority. Process-wide; None = OpenCV's default.
        cv_threads = (config.get("performance") or {}).get("cv_threads")
        if cv_threads is not None:
            cv2.setNumThreads(int(cv_threads))
        self.downscale = float(ocr_cfg["downscale"])
        # Speed options (ocr.speed): each one can be switched off on its own.
        self.speed = dict(ocr_cfg.get("speed") or {})
        # grid_jit: the compiled cell grid search when numba is installed and
        # compiles (logged once by grid_jit.available()); else numpy's.
        if self.speed.get("grid_jit"):
            grid_jit.available()
        self.min_dim = int(ocr_cfg["min_dim"])
        # The image the print is found and read on (ink_channel.py): None =
        # the camera frame as it is (grayscale ink, the standard profile);
        # set = one weighted channel of B, G, R that the whole read uses.
        self.ink_channel = InkChannel.from_config(ocr_cfg.get("ink_channel"))
        self._debug_color = None

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
        # Agreement check for a read accepted with its lowest digit confidence
        # in agreement_band [lo, hi): re-read on agreement_reads slightly moved
        # crops (AGREEMENT_MOVES); accepted only if all of them are accepted
        # with the same code, else NOK unstable_read. 0 reads = off.
        band = val_cfg.get("agreement_band") or [0.7, 0.98]
        self.agreement_band = (float(band[0]), float(band[1]))
        self.agreement_reads = min(len(AGREEMENT_MOVES), int(val_cfg.get("agreement_reads", 2) or 0))
        # Plausibility of line 1 (year digit + day of year, HH:MM) against
        # the capture time (read(capture_time=...)): the print time must lie
        # within [capture - time_window_before, capture + time_window_after]
        # minutes, else NOK implausible_time. Skipped (and logged) when the
        # capture time is unknown or line 1 is another layout.
        self.time_check = bool(val_cfg.get("time_check", False))
        self.time_window_before = int(val_cfg.get("time_window_before", 2))
        self.time_window_after = int(val_cfg.get("time_window_after", 1))
        self._capture_time = None
        self._time_skip_logged = False
        self.skipped_formats = []
        self._last_located = None
        # Second attempt at the print's row angle for a read that failed
        # (see read()); accepted only past these stricter gates.
        # Deskewed crops hold the whole rect also near the frame edge
        # (text_locator.deskew_crop, whole_rect). Off = the crop as before.
        self.deskew_whole_rect = bool(dynamic_roi_cfg.get("deskew_whole_rect", False))
        self.row_angle_retry = bool(dynamic_roi_cfg.get("row_angle_retry", False))
        self.row_angle_retry_min_diff_deg = float(dynamic_roi_cfg.get("row_angle_retry_min_diff_deg", 3.0))
        self.row_angle_retry_min_confidence = float(dynamic_roi_cfg.get("row_angle_retry_min_digit_confidence", 0.9))
        self.row_angle_retry_min_literal = float(dynamic_roi_cfg.get("row_angle_retry_min_literal_score", -0.05))
        # Further retries on other crops (widened along the line, next-best
        # clusters), past the same gates; see _retry_crops. The next-best
        # clusters are off by default: 0 of 7 tried on the test images were
        # accepted.
        self.crop_retry = bool(dynamic_roi_cfg.get("crop_retry", False))
        self.crop_retry_alternatives = int(dynamic_roi_cfg.get("crop_retry_alternatives", 0))
        # Time for the whole retry chain (ms from its start; within the
        # read's own time_budget_ms). Past it the read keeps the first
        # attempt's reject reason. 0 = no limit of its own.
        self.retry_budget_ms = float(dynamic_roi_cfg.get("retry_budget_ms", 800) or 0)
        # Printed ink vs moulded lettering on a retry (ink_retry): the print
        # located again with the locator's ink filter, and every retry
        # deciding "does the crop hold a second line" on printed ink
        # (font_reader.PreparedCrop.print_ink_ratio). None = off.
        self.ink_retry = bool(dynamic_roi_cfg.get("ink_retry", False))
        self.print_ink_ratio = float(dynamic_roi_cfg.get("ink_dark_ratio", 0.6)) if self.ink_retry else None

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
            log.info("OCRead%s: font reader, formats=%s%s", f" [{profile_name}]" if profile_name else "",
                     [f.lines for f in self._read_formats()],
                     f", ink channel {self.ink_channel.describe()}" if self.ink_channel else "")
            for lines in self.skipped_formats:
                log.warning("OCRead: format %s skipped - the font reader model does not support it", lines)

    def warmup(self):
        """One read of a synthetic frame (a dark two-line print on grey), so
        the first real cycles don't pay for first-call costs: OpenCV and
        numpy code paths, kernels, the worker threads (ocr.speed). Called by
        the app at startup when ocr.speed.warmup is set. Returns the time in
        ms; the read itself is thrown away."""
        t = time.perf_counter()
        frame = np.full((480, 744, 3), 170, np.uint8)
        for y, line in ((220, "6274 10:29"), (262, "BB 05/28")):
            cv2.putText(frame, line, (250, y), cv2.FONT_HERSHEY_PLAIN, 2.4, (60, 60, 60), 3)
        debug_dir, self.dynamic_roi_debug_dir = self.dynamic_roi_debug_dir, None
        time_check, self.time_check = self.time_check, False    # a synthetic frame has no capture time
        try:
            self.read(frame)
        except Exception as exc:  # pragma: no cover - a warm-up must never stop the app
            log.warning("OCRead warm-up failed (non-fatal): %s", exc)
        finally:
            self.dynamic_roi_debug_dir = debug_dir
            self.time_check = time_check
        return round((time.perf_counter() - t) * 1000, 1)

    def ink_frame(self, frame):
        """The frame as this engine reads it: its ink channel
        (ocr.ink_channel, ink_channel.py) as a grey BGR image, or the frame
        itself (the same object) without one. read() does this first;
        QC_tools/train_font_reader.py does it before _prepare()."""
        return frame if self.ink_channel is None else self.ink_channel.apply(frame)

    @property
    def validator(self):
        """The first configured format. Assigning one replaces all formats."""
        return self.validators[0]

    @validator.setter
    def validator(self, value):
        self.validators = [value]

    def _read_formats(self):
        return [v for v in self.validators if v.lines not in self.skipped_formats]

    def _read_all_formats(self, crop, deadline=None, prep=None):
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
        the attempts and step_ms so far as its .attempts / .step_ms.
        prep: the crop's PreparedCrop when the caller has one."""
        reads, attempts = [], []
        prep = prep or PreparedCrop(crop, deadline, speed=self.speed)
        # More lines first: once a format is trusted, formats with fewer lines
        # can no longer win and are not read (saves a read per cycle). Once a
        # two-line format is accepted (_check passes) with a clearly right
        # literal score (font reader's early_exit_literal), the other
        # two-line formats are not read either: a print has one line 2, and
        # on the 514 test images no frame had two trusted two-line formats
        # (it saved ~30 ms on 432 of the 502 correct reads). A doubtful
        # accepted read still lets a later format with a higher score win.
        clear = self.font_reader.early_exit_literal
        for v in sorted(self._read_formats(), key=lambda v: -len(v.lines)):
            if any(t and len(x.lines) > len(v.lines) for x, _, t in reads):
                break
            if any(t and len(x.lines) == 2 and (clear is None or r["literal_score"] >= clear)
                   and self._check(x, r) is None for x, r, t in reads):
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

        # speed locate_scale < 1: located on the frame scaled down (the box
        # mapped back; the crop is cut from the full frame). Retries locate
        # on the full frame.
        located = locate_text_region_scaled(frame, dict(self.dynamic_roi_cfg,
                                                        morph_threads=int(self.speed.get("locate_threads", 1))),
                                            float(self.speed.get("locate_scale", 1.0)),
                                            float(self.speed.get("locate_threshold_scale", 1.0)),
                                            debug_dir=self.dynamic_roi_debug_dir, debug_color=self._debug_color)
        self._last_located = located or None
        if not located:
            log.debug("Dynamic ROI: no text region found on this frame - running OCR on the full frame.")
            return frame, None

        cycle_debug_dir = located.get("cycle_debug_dir")
        rotated_rect = located.get("rotated_rect")
        # Extra margin here, beyond dynamic_roi.padding_px (which is already
        # baked into rotated_rect). Changing it changes the crop the font
        # reader sees, so re-check reads on real frames before touching it.
        crop = (deskew_crop(frame, rotated_rect, extra_padding_px=20, whole_rect=self.deskew_whole_rect)
                if rotated_rect else None)
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
        if (verdict["reason"] is None and self.time_check and self._capture_time is not None
                and LINE1.match(r["lines"][0])
                and not plausible_line1(r["lines"][0], self._capture_time, self.time_window_before,
                                        self.time_window_after)):
            return REJECT_IMPLAUSIBLE
        return verdict["reason"]

    def _retry_crops(self, frame, located, first_prep, deadline):
        """The crops a failed read is retried on, in order, as (name,
        PreparedCrop, rect). Every retry reads with the printed-ink line
        count (PreparedCrop.print_ink_ratio): a second band without printed
        ink - embossing such as "P2144/1" beside a one-line print - does not
        refuse the one-line format, which is then read on the printed ink
        alone. A printed line 2, even with dots missing, is ink and still
        refuses it.

        - "row_angle": deskewed at the angle of the print's rows
          (text_locator.row_profile_rect) instead of the minAreaRect angle,
          when they differ by row_angle_retry_min_diff_deg or more. The
          minAreaRect box was tilted 10-50 degrees off a level print on caps
          with a rib or embossing in the ink (33 of the 514 test images).
        - "ink_located": the print located again with the locator's ink
          filter (ink_filter: components without printed ink - embossing -
          dropped before clustering), when that gives another rect. The
          embossing beside a print had joined its cluster and tilted the box.
          Not for the first read: the tighter, level crop it gives read
          worse on 3 of the 514 test images (line 2's grid one cell off on
          two EXP prints) that the first locate reads correctly; after
          "row_angle", whose reads of the first locate came first before.
          Also tried when the first locate found no print (located None).
        - "first_print_ink": the first crop again, read that way (its other
          work is cached; a one-line format is the only read that can
          change).
        - "widened": the located rect lengthened along its line over the
          ink that continues the print (text_locator.extend_rect_along_line):
          a rib between the words of a one-line print left the far word out
          of the cluster, and the crop held only "18:51" of "6286 18:51".
        - "alternative_<i>" or "alternative_<i>_widened": the next-best
          clusters (text_locator.alternative_rects, crop_retry_alternatives
          of them), lengthened like "widened" when there is ink to extend
          over, when the largest cluster was embossing or a moulding defect
          instead of the print. Off by default (crop_retry_alternatives 0).

        Generated lazily, after the previous crop was read: a read accepted
        on an earlier crop costs nothing for the later ones. The first read
        itself is not read this way: on a crop that cut the print the
        printed-ink read can be confident and wrong ("6288 18:37" for 18:33
        at 0.80-0.89 with a wider dot mask), so it is held to the retry
        gates.

        Not retried: each band of a crop on which the one-line format sees
        two lines, read on its own. It read one-line prints with a rib along
        them, but also accepted line 1 of a two-line print whose line 2 was
        printed with dots missing as a one-line code - a rib and a damaged
        line 2 can't be told apart there."""
        padding = int(self.dynamic_roi_cfg.get("padding_px", 10))
        first = located.get("rotated_rect") if located else None

        def prepared(rect):
            crop = deskew_crop(frame, rect, extra_padding_px=20, whole_rect=self.deskew_whole_rect)
            return None if crop is None else PreparedCrop(self._downscale_roi(crop), deadline, self.print_ink_ratio,
                                                          self.speed)

        points = located.get("ink_points") if first is not None else None
        if self.row_angle_retry and points is not None and len(points) >= 3:
            rect = row_profile_rect(points, padding)
            if abs((rect["angle"] - first["angle"] + 90.0) % 180.0 - 90.0) >= self.row_angle_retry_min_diff_deg:
                prep = prepared(rect)
                if prep is not None:
                    yield "row_angle", prep, rect
        if self.ink_retry and (self.crop_retry or located is None):
            ink = locate_text_region(frame, dict(self.dynamic_roi_cfg, ink_filter=True))
            rect = ink["rotated_rect"] if ink else None
            if rect is not None and (first is None or not _same_rect(rect, first)):
                if (prep := prepared(rect)) is not None:
                    yield "ink_located", prep, rect
        if self.print_ink_ratio is not None and first_prep is not None and self.crop_retry:
            first_prep.print_ink_ratio = self.print_ink_ratio
            first_prep.deadline = deadline
            yield "first_print_ink", first_prep, first
        if first is None or not self.crop_retry:       # nothing located, or the axis-aligned box only
            return
        wide = extend_rect_along_line(frame, first, self.dynamic_roi_cfg)
        if wide is not None and (prep := prepared(wide)) is not None:
            yield "widened", prep, wide
        if "_clusters" in located and self.crop_retry_alternatives > 0:
            for i, rect in enumerate(alternative_rects(located, self.crop_retry_alternatives, padding), 1):
                # The widened rect holds the whole cluster too: one read, not two.
                wide = extend_rect_along_line(frame, rect, self.dynamic_roi_cfg)
                name, rect = (f"alternative_{i}", rect) if wide is None else (f"alternative_{i}_widened", wide)
                if (prep := prepared(rect)) is not None:
                    yield name, prep, rect

    def _retry(self, frame, located, first_prep, deadline):
        """Further attempts for a read that failed, on the crops of
        _retry_crops in order, within the same deadline; the first one
        accepted wins. The first attempt is never replaced by this, so a read
        it accepts stays exactly as it was.

        Returns (accepted, attempts): accepted is (validator, read, step_ms,
        name, rect) for the first retry that passes every check of a normal read AND the
        stricter retry gates (row_angle_retry_min_digit_confidence /
        _min_literal_score: a misfitted cell grid can still fit the template
        with one digit guessed, e.g. "EXP 06/28" for "EXP 05/28" at digit
        confidence 0.16, where accepted first reads are >= 0.95 in 99% of
        the test images); otherwise None, also when the deadline passes.
        attempts: those of every crop tried, each marked with its retry
        name, also when none was accepted. rect: the crop's rect in the
        frame. The chain stops at deadline or retry_budget_ms after its
        start, whichever comes first."""
        if self.retry_budget_ms > 0:
            end = time.perf_counter() + self.retry_budget_ms / 1000.0
            deadline = end if deadline is None else min(deadline, end)
        tried, name = [], None
        try:
            for name, prep, rect in self._retry_crops(frame, located, first_prep, deadline):
                validator, r, attempts, step_ms = self._read_all_formats(prep.crop, deadline, prep)
                tried += [dict(a, retry=name) for a in attempts]
                reason = self._check(validator, r)
                if (reason is not None or r["confidence"] < self.row_angle_retry_min_confidence
                        or r["literal_score"] < self.row_angle_retry_min_literal):
                    log.debug("Retry %s not accepted: %s", name,
                              reason or f"literal {r['literal_score']} confidence {r['confidence']}")
                    continue
                return (validator, r, step_ms, name, rect), tried
        except ReadTimeout as exc:
            tried += [dict(a, retry=name) for a in getattr(exc, "attempts", [])]
        return None, tried

    def _agree(self, frame, crop, box, r, retried, deadline):
        """Agreement check of an accepted read r in the grey zone: re-read on
        agreement_reads crops with the read's box moved by AGREEMENT_MOVES
        (cut from the frame like the read's own crop; without a box - the
        axis-aligned fallback - the read's crop itself moved). A retry's
        read is re-read the same way (printed-ink line count). Stops at the
        first extra read that is not accepted or reads another code.

        Returns (reason, info): reason None (all agree), REJECT_UNSTABLE or
        REJECT_TIMEOUT (time budget used up); info: the extra reads' codes
        (None = not accepted) and their time in ms."""
        t = time.perf_counter()
        codes, reason = [], None
        try:
            for dx, dy, da, scale in AGREEMENT_MOVES[:self.agreement_reads]:
                if box is not None:
                    moved = dict(box, center=[box["center"][0] + dx, box["center"][1] + dy], angle=box["angle"] + da,
                                 size=[box["size"][0] * scale, box["size"][1] * scale])
                    c = deskew_crop(frame, moved, extra_padding_px=20, whole_rect=self.deskew_whole_rect)
                    c = None if c is None else self._downscale_roi(c)
                else:
                    # The crop itself moved like the box: turned about its
                    # centre, shifted, and cut scale x as large (the same
                    # pixel size, more border - replicated, as there is no
                    # frame to take it from).
                    h, w = crop.shape[:2]
                    w2, h2 = int(round(w * scale)), int(round(h * scale))
                    m = cv2.getRotationMatrix2D((w / 2, h / 2), da, 1.0)
                    m[:, 2] += ((w2 - w) / 2 - dx, (h2 - h) / 2 - dy)  # the box moves by (dx, dy): the content the other way
                    c = cv2.warpAffine(crop, m, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                code = None
                if c is not None and c.size:
                    prep = PreparedCrop(c, deadline, self.print_ink_ratio if retried else None, self.speed)
                    v2, r2, _, _ = self._read_all_formats(c, deadline, prep)
                    if self._check(v2, r2) is None:
                        code = r2["text"]
                codes.append(code)
                if code != r["text"]:
                    reason = REJECT_UNSTABLE
                    break
        except ReadTimeout:
            reason = REJECT_TIMEOUT
        info = {"codes": codes, "ms": round((time.perf_counter() - t) * 1000, 1)}
        if reason is not None:
            log.debug("Agreement check: %s for %r (extra reads %s)", reason, r["text"], codes)
        return reason, info

    def read(self, frame, profile=False, capture_time=None):
        """
        Read the printed code from the frame.

        OK only when the font reader's own gates (literal_score, digit
        confidence) and the template check (format + value_pattern) all
        pass for the chosen format; "text" is then the canonical code, e.g.
        "4113 08:26 BB 05/26" or "6286 16:24".
        Otherwise failure_reason is format_mismatch / low_confidence /
        invalid_value / no_text_detected / timeout / unstable_read and
        "text" is empty. unstable_read: accepted in the grey zone of digit
        confidence (agreement_band), but slightly moved crops did not all
        read the same code (_agree).

        no_text_detected straight away when the locator (dynamic ROI) found
        no print: reading the full frame instead took 1.2-1.7 s and never
        gave a code (22 of the 514 test images). timeout: the read was still
        running after ocr.read_validation.time_budget_ms.
        implausible_time: line 1 is not a print time near capture_time (a
        datetime: the system clock when the frame was grabbed, or the time in
        a test image's file name; None = unknown, the check is skipped).
        """
        if self.font_reader is None:
            result = {
                "detections": [],
                "text": "",
                "processing_time_ms": 0.0,
                "mode": MODE,
                "error": self.unavailable_reason or "font reader unavailable",
            }
            if self.profile_name is not None:
                result["profile"] = self.profile_name
            return result

        t0 = time.perf_counter()
        # Product profile with an ink channel (ink_channel.py): from here on
        # the frame IS that channel - locate, crops, retries, agreement and
        # the ink maps all use it. The colour frame is kept for the debug
        # capture only. Without one the frame is used as it is.
        self._debug_color = frame if self.ink_channel is not None and self.dynamic_roi_debug_dir else None
        frame = self.ink_frame(frame)
        self._capture_time = capture_time
        if self.time_check and capture_time is None:
            if not self._time_skip_logged:
                log.info("OCRead: capture time unknown - line 1 time plausibility check skipped "
                         "(logged once; per read at debug level)")
                self._time_skip_logged = True
            else:
                log.debug("Capture time unknown: time plausibility check skipped")
        crop, cycle_debug_dir, profile_data, located = self._prepare(frame)
        validator, r, attempts, step_ms, reason = None, None, [], {}, None
        retry_angle, retry_name = None, None
        if located is None and self.dynamic_roi_enabled:
            reason = REJECT_NO_TEXT            # no print found: nothing to read
            if self.ink_retry:                 # ... unless the ink-filtered locate finds one
                deadline = t0 + self.time_budget_ms / 1000.0 if self.time_budget_ms > 0 else None
                retry, attempts = self._retry(frame, None, None, deadline)
                if retry is not None:
                    validator, r, step_ms, retry_name, rect = retry
                    reason = None
        else:
            deadline = t0 + self.time_budget_ms / 1000.0 if self.time_budget_ms > 0 else None
            prep = PreparedCrop(crop, deadline, speed=self.speed)
            try:
                validator, r, attempts, step_ms = self._read_all_formats(crop, deadline, prep)
            except ReadTimeout as exc:
                attempts, step_ms, reason = exc.attempts, exc.step_ms, REJECT_TIMEOUT
            if reason is None:
                reason = self._check(validator, r)
                if reason is not None and (self.row_angle_retry or self.crop_retry) and located is not None:
                    retry, retry_attempts = self._retry(frame, located, prep, deadline)
                    attempts = attempts + retry_attempts
                    if retry is not None:
                        validator, r, step_ms, retry_name, rect = retry
                        retry_angle = rect["angle"] if retry_name == "row_angle" else None
                        reason = None
        agreement = None
        if (reason is None and self.agreement_reads > 0
                and self.agreement_band[0] <= r["confidence"] < self.agreement_band[1]):
            box = rect if retry_name else (located or {}).get("rotated_rect")
            reason, agreement = self._agree(frame, crop, box, r, retry_name is not None, deadline)
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
                        "row_angle_retry": retry_angle, "retry": retry_name, "agreement": agreement,
                        "capture_time": capture_time.isoformat(timespec="seconds") if capture_time else None}
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
        if self.profile_name is not None:
            result["profile"] = self.profile_name
        if profile:
            result["_profile_ms"] = profile_data
        return result
