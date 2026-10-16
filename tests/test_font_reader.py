"""
Tests for the trained dot-font reader (font_reader.py) and its use in OCRead.

The fixtures are real crops from the line: the "6" with a rib through it
that PaddleOCR read as "5", crossing ribs over "3030", and an upside-down
cap. (They were part of the model's training data, so these are regression
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
        from backend.detection.ocr import paddle_ocr

        class NoPaddle:
            def __init__(self, **k): pass
            def ocr(self, *a, **k): raise AssertionError("PaddleOCR must not run in font mode")

        with mock.patch.object(paddle_ocr, "PaddleOCRWorker", NoPaddle):
            e = paddle_ocr.PaddleOCR(app_state=None)
        e.dynamic_roi_debug_dir = None
        e.validator = paddle_ocr.CodeTemplate(TPL, value_pattern=r"^\d{4} ([01]\d|2[0-3]):[0-5]\d BB (0[1-9]|1[0-2])/\d{2}$")
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
        e._prepare = lambda frame, dynamic_roi=False: (crop, None, {}, {"bbox": [300, 250, 460, 330]})  # print WAS found
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


if __name__ == "__main__":
    unittest.main()
