"""
Tests for the trained dot-font reader (font_reader.py) and its use in OCRead.

The fixtures are real crops from the line: the "6" with a rib through it
that a general-purpose OCR read as "5", crossing ribs over "3030", and an upside-down
cap. Single-line prints are the same crops cut off below line 1. (They were
part of the model's training data, so these are regression
tests of the pipeline, not an accuracy measurement - see default.yaml for
the leave-one-bottle-out results.)

Run with:  python -m unittest tests.test_font_reader
"""
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend.detection.ocr.font_reader import FontReader

FIX = Path(__file__).resolve().parent / "fixtures" / "font_reader"
TPL = ["#### ##:##", "BB ##/##"]


class FontReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = FontReader(min_literal_score=-1.0)

    def _read(self, name):
        return self.reader.read(cv2.imread(str(FIX / name)), TPL)

    def test_reads_the_6_under_the_rib(self):
        r = self._read("4113_rib_through_6.png")
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertTrue(r["trusted"])

    def test_reads_through_crossing_ribs(self):
        self.assertEqual(self._read("3030_crossing_ribs.png")["text"], "3030 22:42 BB 02/25")

    def test_reads_an_upside_down_cap(self):
        r = self._read("4113_upside_down.png")
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertTrue(r["flipped"])

    def test_timings_and_upright_early_exit(self):
        r = self._read("4113_rib_through_6.png")
        self.assertIn("timings_ms", r)
        self.assertIn("grid_up_ms", r["timings_ms"])
        self.assertNotIn("grid_down_ms", r["timings_ms"], "clear upright read must skip the upside-down pass")

    def test_upside_down_still_found_with_early_exit(self):
        r = self._read("4113_upside_down.png")
        self.assertTrue(r["flipped"])
        self.assertIn("grid_down_ms", r["timings_ms"])

    def test_blank_crop_is_not_trusted(self):
        r = self.reader.read(np.full((144, 254, 3), 180, np.uint8), TPL)
        self.assertTrue(r is None or not r["trusted"])

    def test_unsupported_template(self):
        self.assertFalse(self.reader.supports(["AAAA ##"]))
        self.assertIsNone(self.reader.read(cv2.imread(str(FIX / "3030_crossing_ribs.png")), ["AAAA ##", "BB ##"]))


class VerificationModelTests(unittest.TestCase):
    def test_a_model_that_never_saw_4113_reads_it(self):
        model = Path(__file__).resolve().parents[1] / "backend/detection/ocr/models/verification/font_reader_without_4113.npz"
        r = FontReader(model, min_literal_score=-1.0).read(cv2.imread(str(FIX / "4113_rib_through_6.png")), TPL)
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")


class _FakeReader:
    min_literal_score, min_digit_confidence = -1.0, 0.0

    def __init__(self, result):
        self.result = result

    def read(self, crop, template):
        return self.result


def _print_found(e):
    """The locator found a print (these tests are about reading it): the
    frame itself is the crop. Without a print OCRead is NOK no_text_detected
    before the reader runs."""
    e._prepare = lambda frame: (frame, None, {}, {"bbox": [0, 0, 1, 1]})
    return e


class OCReadFontModeTests(unittest.TestCase):
    def _engine(self, result, print_found=True):
        from backend.detection.ocread import engine

        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validator = engine.CodeTemplate(TPL, value_pattern=r"^\d{4} ([01]\d|2[0-3]):[0-5]\d BB (0[1-9]|1[0-2])/\d{2}$")
        e.font_reader = _FakeReader(result)
        return _print_found(e) if print_found else e

    def _r(self, text, lit=-0.2, conf=0.99):
        l1, l2 = text[:10], text[11:]
        return {"lines": [l1, l2], "text": text, "literal_score": lit, "confidence": conf, "flipped": False, "angle": 0.0}

    def _frame(self):
        return np.full((480, 744, 3), 190, np.uint8)

    def test_trusted_read_is_ok(self):
        r = self._engine(self._r("4113 08:26 BB 05/26")).read(self._frame())
        self.assertEqual((r["validation"]["accepted"], r["text"]), (True, "4113 08:26 BB 05/26"))
        self.assertNotIn("failure_reason", r)

    def test_garbled_cut_is_rejected(self):
        e = self._engine(self._r("5523 07:12 BB 10/22", lit=-10.6))
        crop = np.full((144, 254, 3), 190, np.uint8)
        e._prepare = lambda frame: (crop, None, {}, {"bbox": [300, 250, 460, 330]})  # print WAS found
        r = e.read(self._frame())
        self.assertEqual((r["failure_reason"], r["text"]), ("format_mismatch", ""))

    def test_nothing_segmented(self):
        self.assertIn(self._engine(None).read(self._frame())["failure_reason"], ("format_mismatch", "no_text_detected"))

    def test_impossible_value_is_rejected(self):
        r = self._engine(self._r("4113 27:26 BB 05/26")).read(self._frame())
        self.assertEqual(r["failure_reason"], "invalid_value")

    def test_rejected_read_shows_no_code(self):
        # The forced decode of a rejected read is not a reading - never display it.
        r = self._engine(self._r("8202 52:34 BB 97/34", lit=-5.6)).read(self._frame())
        self.assertFalse(r["validation"]["accepted"])
        self.assertEqual(r["text"], "")
        self.assertEqual(r["detections"], [])

    def test_no_print_on_the_frame_is_no_text_detected(self):
        r = self._engine(None, print_found=False).read(np.full((480, 744, 3), 128, np.uint8))
        self.assertEqual((r["failure_reason"], r["text"]), ("no_text_detected", ""))

    def test_digit_confidence_gate(self):
        e = self._engine(self._r("4113 08:26 BB 05/26", conf=0.3))
        e.font_reader.min_digit_confidence = 0.5
        self.assertEqual(e.read(self._frame())["failure_reason"], "low_confidence")

    def test_result_contract(self):
        r = self._engine(self._r("4113 08:26 BB 05/26")).read(self._frame(), profile=True)
        self.assertEqual(r["mode"], "ocread")
        for key in ("detections", "text", "processing_time_ms", "validation", "_profile_ms"):
            self.assertIn(key, r)
        self.assertEqual([d["text"] for d in r["detections"]], ["4113 08:26", "BB 05/26"])


class OCReadDefaultConfigTests(unittest.TestCase):
    def test_default_config_loads_the_font_reader(self):
        from backend.detection.ocread.engine import OCRead
        e = OCRead()
        self.assertIsNotNone(e.font_reader, e.unavailable_reason)
        self.assertEqual(e.validator.lines, TPL)


class OCReadUnavailableTests(unittest.TestCase):
    """Without a usable font reader every read is a technical failure, never a guess."""

    def _unavailable_engine(self, template):
        from backend.detection.ocread import engine
        val_cfg = {"template": template, "font_reader": {"model": ""}}
        with mock.patch.dict(engine.cfg["ocr"], {"read_validation": val_cfg}):
            return engine.OCRead()

    def test_unsupported_template_is_an_engine_error(self):
        from backend.core import vision
        e = self._unavailable_engine(["AAAA ##", "BB ##"])
        self.assertIsNone(e.font_reader)
        r = e.read(np.full((480, 744, 3), 190, np.uint8))
        self.assertIn("error", r)
        n = vision._normalize_result(r)
        self.assertEqual((n["status"], n["failure_reason"], n["text"]), ("NOK", "ocr_engine_error", ""))

    def test_missing_model_is_an_engine_error(self):
        from backend.detection.ocread import engine
        val_cfg = {"template": TPL, "font_reader": {"model": "does/not/exist.npz"}}
        with mock.patch.dict(engine.cfg["ocr"], {"read_validation": val_cfg}):
            e = engine.OCRead()
        self.assertIsNone(e.font_reader)
        self.assertIn("could not be loaded", e.read(np.zeros((8, 8, 3), np.uint8))["error"])


# -- several print formats, incl. the single-line one ----------------------------

TPL_EXP = ["#### ##:##", "EXP ##/##"]
TPL_ONE = ["#### ##:##"]
TPL_PROD = ["#### ##:##", "PROD ##/##"]
ALL_FORMATS = [TPL, TPL_EXP, TPL_ONE]
DEFAULT_FORMATS = [TPL, TPL_EXP, TPL_PROD, TPL_ONE]
VALUE_PATTERN = r"^\d{4} ([01]\d|2[0-3]):[0-5]\d( (BB|EXP) (0[1-9]|1[0-2])/\d{2})?$"


def _line1_crop(name, rows):
    """A real single-line print: the fixture cut off below line 1."""
    return cv2.imread(str(FIX / name))[:rows]


class SingleLineReadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = FontReader(min_literal_score=-1.0)

    def test_one_band_is_fitted(self):
        from backend.detection.ocr.font_reader import _fit_lines
        ink = np.zeros((60, 100), np.float32)
        ink[20:40, 10:90] = 1.0
        _, bands = _fit_lines(ink, 1)
        self.assertEqual(len(bands), 1)
        y0, y1 = bands[0]
        self.assertTrue(18 <= y0 <= 22 and 38 <= y1 <= 42, bands)

    def test_reads_a_single_line_print(self):
        r = self.reader.read(_line1_crop("4113_rib_through_6.png", 75), TPL_ONE)
        self.assertEqual((r["text"], r["lines"], r["trusted"]), ("4113 08:26", ["4113 08:26"], True))

    def test_reads_a_single_line_print_through_crossing_ribs(self):
        r = self.reader.read(_line1_crop("3030_crossing_ribs.png", 82), TPL_ONE)
        self.assertEqual(r["text"], "3030 22:42")
        self.assertTrue(r["trusted"])

    def test_reads_an_upside_down_single_line_print(self):
        r = self.reader.read(_line1_crop("4113_rib_through_6.png", 75)[::-1, ::-1], TPL_ONE)
        self.assertEqual(r["text"], "4113 08:26")
        self.assertTrue(r["flipped"])

    def test_two_line_print_is_not_read_as_one_line(self):
        for name in ("4113_rib_through_6.png", "3030_crossing_ribs.png", "4113_upside_down.png"):
            with self.subTest(name):
                self.assertIsNone(self.reader.read(cv2.imread(str(FIX / name)), TPL_ONE))

    def test_line_2_alone_is_not_trusted_as_one_line(self):
        r = self.reader.read(cv2.imread(str(FIX / "4113_rib_through_6.png"))[74:], TPL_ONE)
        self.assertTrue(r is None or not r["trusted"])

    def test_formats_with_unknown_letters_are_not_supported(self):
        self.assertTrue(self.reader.supports(TPL_ONE))
        if "E" not in self.reader.idx:
            self.assertFalse(self.reader.supports(TPL_EXP))
            self.assertIsNone(self.reader.read(cv2.imread(str(FIX / "3030_crossing_ribs.png")), TPL_EXP))


class _PerFormatReader:
    """Fake font reader: a fixed result per template (None = nothing read)."""
    min_literal_score, min_digit_confidence = -1.0, 0.0

    def __init__(self, results):
        self.results = {tuple(k): v for k, v in results.items()}
        self.calls = []

    def read(self, crop, template):
        self.calls.append(list(template))
        return self.results.get(tuple(template))


def _res(*lines, lit=-0.1, conf=0.99):
    return {"lines": list(lines), "text": " ".join(lines), "literal_score": lit, "confidence": conf,
            "flipped": False, "angle": 0.0}


class _SequenceReader(_PerFormatReader):
    """Fake font reader: reads `first` per template on the first crop,
    `retry` on any later crop (the row angle retry)."""

    def __init__(self, first, retry):
        super().__init__(first)
        self.retry = {tuple(k): v for k, v in retry.items()}
        self.crops = []

    def read(self, crop, template):
        if not any(c is crop for c in self.crops):
            self.crops.append(crop)
        self.calls.append(list(template))
        return (self.results if self.crops[0] is crop else self.retry).get(tuple(template))


class OCReadRowAngleRetryTests(unittest.TestCase):
    """A failed read is retried once at the print's row angle; the retry is
    accepted only past the normal checks and the stricter retry gates, and
    an accepted first read is never retried."""

    def _engine(self, first, retry, box_angle=35.0):
        from backend.detection.ocread import engine
        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validators = [engine.CodeTemplate(f, value_pattern=VALUE_PATTERN) for f in (TPL, TPL_ONE)]
        e.skipped_formats = []
        e.row_angle_retry = True
        e.row_angle_retry_min_diff_deg = 3.0
        e.row_angle_retry_min_confidence, e.row_angle_retry_min_literal = 0.9, -0.05
        e.font_reader = _SequenceReader(first, retry)
        # A level two-line print (rows of ink at y=200 and y=230) whose box
        # the locator tilted by box_angle.
        xs = np.arange(250, 500, 3.0)
        pts = np.vstack([np.c_[xs, np.full_like(xs, y)] for y in (199, 200, 201, 229, 230, 231)]).astype(np.float32)
        located = {"bbox": [250, 190, 500, 240], "ink_points": pts,
                   "rotated_rect": {"center": [375.0, 215.0], "size": [300.0, 90.0], "angle": box_angle}}
        e._prepare = lambda frame: (frame[150:300, 200:550], None, {}, located)
        return e

    def _frame(self):
        return np.full((480, 744, 3), 190, np.uint8)

    def test_failed_read_is_accepted_from_the_retry(self):
        e = self._engine({tuple(TPL): _res("6288 13:19", "BB 88/88", lit=-6.0)},
                         {tuple(TPL): _res("6288 13:19", "BB 05/28", lit=0.0, conf=1.0)})
        r = e.read(self._frame())
        self.assertEqual(r["text"], "6288 13:19 BB 05/28")
        summary = r["validation"]["reads"][0]
        self.assertAlmostEqual(summary["row_angle_retry"], 0.0, delta=0.5)
        self.assertTrue(any(a.get("retry") for a in summary["formats_tried"]))

    def test_accepted_first_read_is_not_retried(self):
        e = self._engine({tuple(TPL): _res("6288 13:19", "BB 05/28")},
                         {tuple(TPL): _res("6288 13:19", "BB 06/28", lit=0.0, conf=1.0)})
        r = e.read(self._frame())
        self.assertEqual(r["text"], "6288 13:19 BB 05/28")
        self.assertIsNone(r["validation"]["reads"][0]["row_angle_retry"])
        self.assertEqual(len(e.font_reader.crops), 1)

    def test_retry_below_the_digit_confidence_gate_stays_nok(self):
        # 15-10-2026_5-58-43_122: the retry's line 2 cells were half a cell
        # off and "05/28" fit the template as "06/28" at confidence 0.16.
        e = self._engine({tuple(TPL): _res("6288 05:58", "BB 88/88", lit=-6.0)},
                         {tuple(TPL): _res("6288 05:58", "BB 06/28", lit=0.0, conf=0.161)})
        r = e.read(self._frame())
        self.assertEqual((r["failure_reason"], r["text"]), ("format_mismatch", ""))
        self.assertIsNone(r["validation"]["reads"][0]["row_angle_retry"])

    def test_retry_below_the_literal_gate_stays_nok(self):
        e = self._engine({tuple(TPL): _res("6288 05:58", "BB 88/88", lit=-6.0)},
                         {tuple(TPL): _res("6288 05:58", "BB 05/28", lit=-0.4, conf=1.0)})
        self.assertEqual(e.read(self._frame())["failure_reason"], "format_mismatch")

    def test_no_retry_when_the_row_angle_matches_the_box(self):
        e = self._engine({tuple(TPL): _res("6288 13:19", "BB 88/88", lit=-6.0)},
                         {tuple(TPL): _res("6288 13:19", "BB 05/28", lit=0.0, conf=1.0)}, box_angle=1.0)
        self.assertEqual(e.read(self._frame())["failure_reason"], "format_mismatch")
        self.assertEqual(len(e.font_reader.crops), 1)


class OCReadTwoLineGuardTests(unittest.TestCase):
    """Real frames of two-line prints that must never come out as a one-line
    code (line 2 ignored), with the production config and retries:

    - two_line_bb_line2_dots_missing (1-10-2026_19-30-54): "6274 19:31" /
      "BB 05/28" with line 2 printed with dots missing - a misprint, NOK.
      A retry that read each line band on its own accepted "6274 19:31".
    - two_line_bb_rib_across (15-10-2026_14-56-21) and
      two_line_exp_rib_across (15-10-2026_5-58-43): "BB 05/28" and
      "EXP 05/28" prints with a rib across them; the same retry accepted
      line 1 of the first as "6288 14:56".

    No time budget, so a NOK here is never just a timeout."""

    FRAMES = FIX.parent / "ocread"

    @classmethod
    def setUpClass(cls):
        import logging
        from backend.detection.ocread.engine import OCRead
        from QC_tools.evaluate_images import load_image
        logging.disable(logging.WARNING)
        cls.engine = OCRead()
        cls.engine.dynamic_roi_debug_dir = None
        cls.engine.time_budget_ms = 0
        cls.load = staticmethod(load_image)

    @classmethod
    def tearDownClass(cls):
        import logging
        logging.disable(logging.NOTSET)

    def _read(self, name):
        return self.engine.read(self.load(self.FRAMES / name, 744))

    def test_line_2_with_dots_missing_stays_nok(self):
        r = self._read("two_line_bb_line2_dots_missing.jpg")
        self.assertFalse(r["validation"]["accepted"])
        self.assertEqual(r["text"], "")

    def test_two_line_prints_with_a_rib_are_never_read_as_one_line(self):
        for name in ("two_line_bb_rib_across.jpg", "two_line_exp_rib_across.jpg"):
            r = self._read(name)
            if r["validation"]["accepted"]:
                self.assertEqual(len(r["validation"]["format"]), 2, f"{name}: read as {r['text']!r}")


class OCReadFormatSelectionTests(unittest.TestCase):
    def _engine(self, results, formats=ALL_FORMATS, skipped=()):
        from backend.detection.ocread import engine
        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validators = [engine.CodeTemplate(f, value_pattern=VALUE_PATTERN) for f in formats]
        e.skipped_formats = [list(f) for f in skipped]
        e.font_reader = _PerFormatReader(results)
        return _print_found(e)

    def _frame(self):
        return np.full((480, 744, 3), 190, np.uint8)

    def test_two_line_beats_single_line_when_both_fit(self):
        r = self._engine({tuple(TPL): _res("6274 10:29", "BB 05/28", lit=-0.4),
                          tuple(TPL_ONE): _res("6274 10:29", lit=-0.01)}).read(self._frame())
        self.assertEqual(r["text"], "6274 10:29 BB 05/28")
        self.assertEqual(r["validation"]["format"], TPL)

    def test_single_line_format_is_not_read_once_a_two_line_format_is_trusted(self):
        e = self._engine({tuple(TPL): _res("6274 10:29", "BB 05/28"),
                          tuple(TPL_ONE): _res("6274 10:29")}, formats=[TPL_ONE, TPL])
        self.assertEqual(e.read(self._frame())["text"], "6274 10:29 BB 05/28")
        self.assertEqual(e.font_reader.calls, [TPL])

    def test_trusted_two_line_wins_even_when_its_value_is_invalid(self):
        # Line 2 was read and is impossible - that must not turn into an OK
        # single-line read of line 1.
        r = self._engine({tuple(TPL): _res("6274 10:29", "BB 13/28"),
                          tuple(TPL_ONE): _res("6274 10:29")}).read(self._frame())
        self.assertEqual((r["failure_reason"], r["text"]), ("invalid_value", ""))
        self.assertEqual(r["validation"]["format"], TPL)

    def test_single_line_print_is_read_as_the_single_line_format(self):
        r = self._engine({tuple(TPL): _res("6286 16:24", "BB 88/88", lit=-6.0),
                          tuple(TPL_EXP): None,
                          tuple(TPL_ONE): _res("6286 16:24")}).read(self._frame())
        self.assertTrue(r["validation"]["accepted"])
        self.assertEqual(r["text"], "6286 16:24")
        self.assertEqual(r["validation"]["format"], TPL_ONE)
        self.assertEqual([d["text"] for d in r["detections"]], ["6286 16:24"])

    def test_exp_format(self):
        r = self._engine({tuple(TPL): _res("6274 05:58", "BB 05/28", lit=-4.0),
                          tuple(TPL_EXP): _res("6274 05:58", "EXP 05/28", lit=-0.2)}).read(self._frame())
        self.assertEqual(r["text"], "6274 05:58 EXP 05/28")
        self.assertEqual(r["validation"]["format"], TPL_EXP)

    def test_higher_literal_score_wins_between_formats_with_equal_lines(self):
        r = self._engine({tuple(TPL): _res("6274 05:58", "BB 05/28", lit=-0.9),
                          tuple(TPL_EXP): _res("6274 05:58", "EXP 05/28", lit=-0.1)}).read(self._frame())
        self.assertEqual(r["validation"]["format"], TPL_EXP)

    def test_unsupported_format_is_skipped(self):
        e = self._engine({tuple(TPL_ONE): _res("6286 16:24")}, skipped=[TPL_EXP])
        r = e.read(self._frame())
        self.assertEqual(r["text"], "6286 16:24")
        self.assertEqual(e.font_reader.calls, [TPL, TPL_ONE])

    def test_no_trusted_read_shows_no_code(self):
        r = self._engine({tuple(TPL): _res("8202 52:34", "BB 97/34", lit=-5.6),
                          tuple(TPL_ONE): _res("8202 52:34", lit=-3.0)}).read(self._frame())
        self.assertIn(r["failure_reason"], ("format_mismatch", "no_text_detected"))
        self.assertEqual((r["text"], r["detections"]), ("", []))
        self.assertFalse(r["validation"]["accepted"])

    def test_tried_formats_are_in_the_read_summary(self):
        r = self._engine({tuple(TPL_ONE): _res("6286 16:24")}).read(self._frame())
        tried = r["validation"]["reads"][0]["formats_tried"]
        self.assertEqual([t["format"] for t in tried], ALL_FORMATS)
        self.assertEqual([t["trusted"] for t in tried], [False, False, True])


class OCReadRealFormatTests(unittest.TestCase):
    """The real font reader on real crops, with all three formats configured."""

    @classmethod
    def setUpClass(cls):
        from backend.detection.ocread import engine
        val_cfg = {"templates": ALL_FORMATS, "value_pattern": VALUE_PATTERN, "font_reader": {"model": ""}}
        with mock.patch.dict(engine.cfg["ocr"], {"read_validation": val_cfg}):
            cls.engine = engine.OCRead()
        cls.engine.dynamic_roi_debug_dir = None

    def _read(self, crop):
        self.engine._prepare = lambda frame: (crop, None, {}, {"bbox": [0, 0, 1, 1]})
        return self.engine.read(np.zeros((8, 8, 3), np.uint8))

    def test_two_line_print(self):
        r = self._read(cv2.imread(str(FIX / "4113_rib_through_6.png")))
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["validation"]["format"], TPL)

    def test_single_line_print(self):
        r = self._read(_line1_crop("3030_crossing_ribs.png", 82))
        self.assertEqual(r["text"], "3030 22:42")
        self.assertEqual(r["validation"]["format"], TPL_ONE)


class TemplatesConfigTests(unittest.TestCase):
    def test_templates_list(self):
        from backend.detection.ocr.code_validator import CodeTemplate
        vs = CodeTemplate.list_from_config({"templates": ALL_FORMATS, "value_pattern": VALUE_PATTERN})
        self.assertEqual([v.lines for v in vs], ALL_FORMATS)
        self.assertTrue(all(v.value_pattern.pattern == VALUE_PATTERN for v in vs))

    def test_old_single_template_key_still_works(self):
        from backend.detection.ocr.code_validator import CodeTemplate
        from backend.detection.ocread import engine
        self.assertEqual([v.lines for v in CodeTemplate.list_from_config({"template": TPL})], [TPL])
        val_cfg = {"template": TPL, "font_reader": {"model": ""}}
        with mock.patch.dict(engine.cfg["ocr"], {"read_validation": val_cfg}):
            e = engine.OCRead()
        self.assertIsNotNone(e.font_reader, e.unavailable_reason)
        self.assertEqual([v.lines for v in e.validators], [TPL])

    def test_templates_wins_over_template(self):
        from backend.detection.ocr.code_validator import CodeTemplate
        vs = CodeTemplate.list_from_config({"template": TPL, "templates": [TPL_ONE]})
        self.assertEqual([v.lines for v in vs], [TPL_ONE])

    def test_default_config_has_the_four_formats(self):
        from backend.detection.ocread.engine import OCRead
        e = OCRead()
        self.assertEqual([v.lines for v in e.validators], DEFAULT_FORMATS)
        self.assertIsNotNone(e.font_reader, e.unavailable_reason)
        reader = FontReader()
        self.assertEqual(e.skipped_formats, [f for f in (TPL_EXP, TPL_PROD) if not reader.supports(f)])

    def test_value_pattern_accepts_all_formats(self):
        import re
        from backend.core.config_loader import cfg
        pat = re.compile(cfg["ocr"]["read_validation"]["value_pattern"])
        for ok in ("6274 10:29 BB 05/28", "6274 05:58 EXP 05/28", "6288 22:00 PROD 10/26", "6286 16:24"):
            self.assertTrue(pat.fullmatch(ok), ok)
        for bad in ("6286 24:00", "6274 10:29 BB 13/28", "6274 10:29 XX 05/28", "6274 10:29 BB",
                    "6288 22:00 PROD 13/26", "6288 24:00 PROD 10/26", "6288 22:00 PR0D 10/26"):
            self.assertFalse(pat.fullmatch(bad), bad)


class LetterOIsDigitZeroTests(unittest.TestCase):
    """In this dot font O and 0 are the same glyph: a literal O is scored with
    the 0 class. Digit positions are never mapped."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.reader = FontReader(min_literal_score=-1.0)
        # The current model may lack P, R and D: add them (never predicted) so
        # the PROD format is supported, as it will be after retraining.
        d = np.load(str(Path(__file__).resolve().parents[1] / "backend/detection/ocr/models/font_reader.npz"))
        n = d["W1"].shape[1]
        extra = [c for c in "PRD" if c not in list(d["classes"])]
        cls.tmp = tempfile.TemporaryDirectory()
        model = Path(cls.tmp.name) / "font_reader_prd.npz"
        np.savez(model, W1=d["W1"], b1=d["b1"],
                 W2=np.hstack([d["W2"], np.zeros((n, len(extra)), np.float32)]),
                 b2=np.r_[d["b2"], np.full(len(extra), -50.0, np.float32)],
                 classes=np.array(list(d["classes"]) + extra))
        cls.reader_rd = FontReader(model, min_literal_score=-1.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prod_needs_r_and_d(self):
        self.assertTrue(self.reader_rd.supports(TPL_PROD))
        if "R" not in self.reader.idx:
            self.assertFalse(self.reader.supports(TPL_PROD))

    def test_literal_o_is_read_with_the_0_class_on_a_real_crop(self):
        # "BB 02/25" printed; template with a literal O where the 0 is.
        r = self.reader.read(cv2.imread(str(FIX / "3030_crossing_ribs.png")), ["#### ##:##", "BB O#/##"])
        self.assertEqual(r["text"], "3030 22:42 BB O2/25")
        self.assertTrue(r["trusted"])

    def test_prod_decode(self):
        rd = self.reader_rd
        text = "628822:00PROD10/26"                    # cells in reading order
        probs = np.full((len(text), len(rd.classes)), 1e-4)
        for i, ch in enumerate(text):
            probs[i, rd.idx["0" if ch == "O" else ch]] = 1.0
        lines, lit, conf = rd._decode(probs, TPL_PROD)
        self.assertEqual(lines, ["6288 22:00", "PROD 10/26"])
        self.assertGreater(lit, -0.01)                 # the O cell scored as 0 -> a clear literal
        self.assertGreater(conf, 0.99)

    def test_digit_positions_are_never_mapped(self):
        rd = self.reader_rd
        # The digits 0 stay digits; the "O" scored at the literal position
        # only ever comes from the template.
        probs = np.full((18, len(rd.classes)), 1e-4)
        for i, ch in enumerate("000000:00PROD00/00"):
            probs[i, rd.idx["0" if ch == "O" else ch]] = 1.0
        lines, _, _ = rd._decode(probs, TPL_PROD)
        self.assertEqual(lines, ["0000 00:00", "PROD 00/00"])

    def test_a_prod_read_is_validated_and_accepted(self):
        from backend.detection.ocread import engine
        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validators = [engine.CodeTemplate(f, value_pattern=(
            r"^\d{4} ([01]\d|2[0-3]):[0-5]\d( (BB|EXP|PROD) (0[1-9]|1[0-2])/\d{2})?$")) for f in DEFAULT_FORMATS]
        e.skipped_formats = []
        e.font_reader = _PerFormatReader({tuple(TPL_PROD): _res("6288 22:00", "PROD 10/26")})
        _print_found(e)
        r = e.read(np.full((480, 744, 3), 190, np.uint8))
        self.assertEqual(r["text"], "6288 22:00 PROD 10/26")
        self.assertEqual(r["validation"]["format"], TPL_PROD)


# -- shared work between formats (PreparedCrop) and the vectorised grid fit -----

def _fit_cells_per_pitch(fit, band, template_line, near_pitch=None):
    """_fit_cells() as it was before the pitches were vectorised: one pitch
    at a time. Reference for the vectorised version (must be identical)."""
    from backend.detection.ocr.font_reader import LINE_PITCH_TOLERANCE, PITCH_RANGE, _layouts
    y0, y1 = band; h = max(1, y1 - y0)
    p_lo, p_hi = PITCH_RANGE[0] * h, PITCH_RANGE[1] * h
    if near_pitch:
        p_lo = max(p_lo, near_pitch / LINE_PITCH_TOLERANCE)
        p_hi = min(p_hi, near_pitch * LINE_PITCH_TOLERANCE)
        if p_lo >= p_hi:
            p_lo, p_hi = PITCH_RANGE[0] * h, PITCH_RANGE[1] * h
    col = fit[y0:y1].sum(axis=0) / h; W = len(col)
    cs = np.r_[0.0, np.cumsum(col)]

    def mean(a, b):
        a = np.clip(np.round(a).astype(int), 0, W); b = np.clip(np.round(b).astype(int), 0, W)
        return np.where(b > a, (cs[b] - cs[a]) / np.maximum(1, b - a), 0.0)

    best = (-1e18, None)
    for cells, labels in _layouts(template_line):
        K = len(cells)
        for p in np.arange(p_lo, p_hi, 0.25):
            x0 = np.arange(-0.5 * p, W - K * p * 0.8, 0.5)
            if len(x0) == 0:
                continue
            s = np.zeros_like(x0)
            for k, kind in enumerate(cells):
                a = x0 + k * p
                core = mean(a + 0.15 * p, a + 0.85 * p)
                s += core if kind == "G" else -2.0 * core
                s -= 0.5 * mean(a - 0.08 * p, a + 0.08 * p)
            s -= 2.0 * (mean(x0 - p, x0) + mean(x0 + K * p, x0 + (K + 1) * p))
            s /= K
            i = int(np.argmax(s))
            if s[i] > best[0]:
                best = (float(s[i]), (labels, [(float(x0[i] + k * p), float(x0[i] + (k + 1) * p)) for k in range(K)]))
    return best


class SharedFormatWorkTests(unittest.TestCase):
    """Several formats on one crop share the format-independent work; the
    results must be exactly those of each format read on its own."""

    @classmethod
    def setUpClass(cls):
        cls.reader = FontReader(min_literal_score=-1.0)
        cls.formats = [f for f in DEFAULT_FORMATS if cls.reader.supports(f)]

    def _crops(self):
        for name in ("4113_rib_through_6.png", "3030_crossing_ribs.png", "4113_upside_down.png"):
            yield name, cv2.imread(str(FIX / name))
        yield "single line", _line1_crop("3030_crossing_ribs.png", 82)

    def test_prepared_crop_reads_like_each_format_on_its_own(self):
        from backend.detection.ocr.font_reader import PreparedCrop
        strip = lambda r: r and {k: v for k, v in r.items() if k != "timings_ms"}
        for name, crop in self._crops():
            prep = PreparedCrop(crop)
            for f in self.formats:
                with self.subTest(crop=name, format=f):
                    self.assertEqual(strip(self.reader.read(prep, f)), strip(self.reader.read(crop, f)))
            self.assertGreater(prep.timings_ms["cell_fit"], 0)

    def test_line_1_grid_is_fitted_once_for_all_formats(self):
        from backend.detection.ocr import font_reader
        from backend.detection.ocr.font_reader import PreparedCrop
        prep = PreparedCrop(cv2.imread(str(FIX / "4113_rib_through_6.png")))
        calls = []
        real = font_reader._fit_cells
        with mock.patch.object(font_reader, "_fit_cells", lambda *a: calls.append(a[2]) or real(*a)):
            for f in self.formats:
                prep.segment(f, False)
        self.assertEqual(calls.count("#### ##:##"), 2)     # once on 2 bands, once on 1 band
        self.assertEqual(len(calls), 2 + len([f for f in self.formats if len(f) == 2]))

    def test_vectorised_grid_fit_matches_the_per_pitch_loop(self):
        from backend.detection.ocr.font_reader import _fit_cells, _fit_lines, ink_maps, line_pitch
        rng = np.random.default_rng(0)
        cases = []
        for name, crop in self._crops():
            raw = ink_maps(crop)[1]
            _, bands = _fit_lines(raw, 2)
            l1 = _fit_cells_per_pitch(raw, bands[0], "#### ##:##")
            cases += [(raw, bands[0], "#### ##:##", None)] + [
                (raw, bands[1], f[1], line_pitch(l1)) for f in self.formats if len(f) == 2]
        for _ in range(10):                                 # random ink, odd sizes and pitch hints
            H, W = int(rng.integers(20, 120)), int(rng.integers(10, 400))
            y0 = int(rng.integers(0, H - 5))
            cases.append((rng.random((H, W)).astype(np.float32), (y0, int(min(H, y0 + rng.integers(1, 40)))),
                          "BB ##/##", None if rng.random() < 0.5 else float(rng.uniform(5, 40))))
        for fit, band, tl, near in cases:
            with self.subTest(band=band, line=tl, near=near):
                self.assertEqual(_fit_cells(fit, band, tl, near), _fit_cells_per_pitch(fit, band, tl, near))

    def test_engine_reports_time_per_step_and_per_format(self):
        from backend.detection.ocread import engine
        val_cfg = {"templates": ALL_FORMATS, "value_pattern": VALUE_PATTERN, "font_reader": {"model": ""}}
        with mock.patch.dict(engine.cfg["ocr"], {"read_validation": val_cfg}):
            e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        crop = cv2.imread(str(FIX / "4113_rib_through_6.png"))
        e._prepare = lambda frame: (crop, None, {"roi_ms": 12.0}, {"bbox": [0, 0, 1, 1]})
        read = e.read(np.zeros((8, 8, 3), np.uint8))["validation"]["reads"][0]
        self.assertEqual(read["step_ms"]["locate"], 12.0)
        for step in ("ink", "deskew", "band_fit", "cell_fit", "cells"):
            self.assertIn(step, read["step_ms"])
        self.assertTrue(all(t["ms"] > 0 for t in read["formats_tried"]))


# -- no print found / time budget ---------------------------------------------------

class OCReadNoPrintAndTimeBudgetTests(unittest.TestCase):
    def _engine(self, results=None, **cfg_overrides):
        from backend.detection.ocread import engine
        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validators = [engine.CodeTemplate(TPL, value_pattern=VALUE_PATTERN)]
        e.skipped_formats = []
        for k, v in cfg_overrides.items():
            setattr(e, k, v)
        if results is not None:
            e.font_reader = _PerFormatReader(results)
        return e

    def test_no_print_found_is_nok_without_reading(self):
        e = self._engine({tuple(TPL): _res("6274 10:29", "BB 05/28")})
        r = e.read(np.full((480, 744, 3), 190, np.uint8))      # the locator finds nothing here
        self.assertEqual((r["failure_reason"], r["text"]), ("no_text_detected", ""))
        self.assertEqual(e.font_reader.calls, [], "the full frame must not be read")

    def test_without_dynamic_roi_the_full_frame_is_still_read(self):
        e = self._engine({tuple(TPL): _res("6274 10:29", "BB 05/28")}, dynamic_roi_enabled=False)
        r = e.read(np.full((480, 744, 3), 190, np.uint8))
        self.assertEqual(r["text"], "6274 10:29 BB 05/28")

    def test_default_time_budget(self):
        self.assertEqual(self._engine().time_budget_ms, 1500.0)

    def test_used_up_time_budget_is_nok_timeout_without_a_code(self):
        from backend.core import vision
        e = _print_found(self._engine(time_budget_ms=0.001))
        crop = cv2.imread(str(FIX / "4113_rib_through_6.png"))
        e._prepare = lambda frame: (crop, None, {}, {"bbox": [0, 0, 1, 1]})
        r = e.read(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual((r["failure_reason"], r["text"], r["detections"]), ("timeout", "", []))
        read = r["validation"]["reads"][0]
        self.assertFalse(read["accepted"])
        self.assertTrue(read["formats_tried"][-1]["timeout"])
        n = vision._normalize_result(r)
        self.assertEqual((n["status"], n["failure_reason"], n["text"]), ("NOK", "timeout", ""))

    def test_no_budget_reads_normally(self):
        e = self._engine(time_budget_ms=0.0)
        crop = cv2.imread(str(FIX / "4113_rib_through_6.png"))
        e._prepare = lambda frame: (crop, None, {}, {"bbox": [0, 0, 1, 1]})
        self.assertEqual(e.read(np.zeros((8, 8, 3), np.uint8))["text"], "4113 08:26 BB 05/26")

    def test_prepared_crop_stops_after_its_deadline(self):
        import time
        from backend.detection.ocr.font_reader import PreparedCrop, ReadTimeout
        prep = PreparedCrop(cv2.imread(str(FIX / "4113_rib_through_6.png")), deadline=time.perf_counter() - 1)
        with self.assertRaises(ReadTimeout):
            FontReader().read(prep, TPL)


if __name__ == "__main__":
    unittest.main()
