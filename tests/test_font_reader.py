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


class OCReadFontModeTests(unittest.TestCase):
    def _engine(self, result):
        from backend.detection.ocread import engine

        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validator = engine.CodeTemplate(TPL, value_pattern=r"^\d{4} ([01]\d|2[0-3]):[0-5]\d BB (0[1-9]|1[0-2])/\d{2}$")
        e.font_reader = _FakeReader(result)
        return e

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
        r = self._engine(None).read(np.full((480, 744, 3), 128, np.uint8))
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
ALL_FORMATS = [TPL, TPL_EXP, TPL_ONE]
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


class OCReadFormatSelectionTests(unittest.TestCase):
    def _engine(self, results, formats=ALL_FORMATS, skipped=()):
        from backend.detection.ocread import engine
        e = engine.OCRead()
        e.dynamic_roi_debug_dir = None
        e.validators = [engine.CodeTemplate(f, value_pattern=VALUE_PATTERN) for f in formats]
        e.skipped_formats = [list(f) for f in skipped]
        e.font_reader = _PerFormatReader(results)
        return e

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

    def test_default_config_has_the_three_formats(self):
        from backend.detection.ocread.engine import OCRead
        e = OCRead()
        self.assertEqual([v.lines for v in e.validators], ALL_FORMATS)
        self.assertIsNotNone(e.font_reader, e.unavailable_reason)
        expected_skipped = [] if FontReader().supports(TPL_EXP) else [TPL_EXP]
        self.assertEqual(e.skipped_formats, expected_skipped)

    def test_value_pattern_accepts_all_formats(self):
        import re
        from backend.core.config_loader import cfg
        pat = re.compile(cfg["ocr"]["read_validation"]["value_pattern"])
        for ok in ("6274 10:29 BB 05/28", "6274 05:58 EXP 05/28", "6286 16:24"):
            self.assertTrue(pat.fullmatch(ok), ok)
        for bad in ("6286 24:00", "6274 10:29 BB 13/28", "6274 10:29 XX 05/28", "6274 10:29 BB"):
            self.assertFalse(pat.fullmatch(bad), bad)


if __name__ == "__main__":
    unittest.main()
