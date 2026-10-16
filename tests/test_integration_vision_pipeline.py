import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core import vision
from backend.output import result_writer


class _FakeAppState:
    def get_vision_mode(self):
        return "ocread"


class _FakeOCR:
    def __init__(self, response):
        self._response = response

    def read(self, _frame, profile=False):
        return dict(self._response)


_GOOD_READ = {
    "detections": [{"text": "4113 08:26", "confidence": 0.97}, {"text": "BB 05/26", "confidence": 0.97}],
    "text": "4113 08:26 BB 05/26",
    "processing_time_ms": 2.1,
    "mode": "ocread",
    "validation": {"accepted": True, "reason": None, "reads": [], "fusion": None},
}


class VisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._old_app_state = vision._app_state
        self._old_ocr = vision.ocr_instance
        vision._app_state = _FakeAppState()

    def tearDown(self):
        vision._app_state = self._old_app_state
        vision.ocr_instance = self._old_ocr

    def test_run_vision_accepted_read_is_ok(self):
        vision.ocr_instance = _FakeOCR(_GOOD_READ)

        result = vision.run_vision(frame=object())

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["mode"], "ocread")
        self.assertEqual(result["text"], "4113 08:26 BB 05/26")
        self.assertAlmostEqual(result["confidence"], 0.97)

    def test_run_vision_rejected_read_is_nok_with_reason(self):
        vision.ocr_instance = _FakeOCR({
            "detections": [], "text": "", "processing_time_ms": 1.0, "mode": "ocread",
            "validation": {"accepted": False, "reason": "invalid_value", "reads": [], "fusion": None},
        })

        result = vision.run_vision(frame=object())
        self.assertEqual((result["status"], result["failure_reason"]), ("NOK", "invalid_value"))

    def test_run_vision_returns_nok_for_missing_frame(self):
        vision.ocr_instance = _FakeOCR(_GOOD_READ)

        result = vision.run_vision(frame=None)

        self.assertEqual(result["status"], "NOK")
        self.assertEqual(result["error"], "no_frame")
        self.assertEqual(result["detections"], [])

    def test_run_vision_callback_path_returns_normalized_payload(self):
        vision.ocr_instance = _FakeOCR(_GOOD_READ)

        event = threading.Event()
        captured = {}

        def callback(result):
            captured.update(result)
            event.set()

        vision.run_vision(frame=object(), callback=callback)
        self.assertTrue(event.wait(2.0), "vision callback did not complete in time")

        self.assertEqual(captured["status"], "OK")
        self.assertEqual(captured["mode"], "ocread")
        self.assertEqual(len(captured["detections"]), 2)
        self.assertEqual(captured["text"], "4113 08:26 BB 05/26")

    def test_run_vision_callback_path_reports_engine_exception(self):
        class _Broken:
            def read(self, _frame, profile=False):
                raise RuntimeError("boom")

        vision.ocr_instance = _Broken()
        event = threading.Event()
        captured = {}

        def callback(result):
            captured.update(result)
            event.set()

        vision.run_vision(frame=object(), callback=callback)
        self.assertTrue(event.wait(2.0), "vision callback did not complete in time")
        self.assertEqual((captured["status"], captured["failure_reason"]), ("NOK", "ocr_engine_error"))

    def test_run_vision_callback_path_returns_nok_for_missing_frame(self):
        vision.ocr_instance = _FakeOCR(_GOOD_READ)

        event = threading.Event()
        captured = {}

        def callback(result):
            captured.update(result)
            event.set()

        vision.run_vision(frame=None, callback=callback)
        self.assertTrue(event.wait(2.0), "vision callback did not complete in time")

        self.assertEqual(captured["status"], "NOK")
        self.assertEqual(captured["error"], "no_frame")


class ResultWriterIntegrationTests(unittest.TestCase):
    def test_save_result_writes_jsonl_with_timestamps(self):
        sample = {"status": "OK", "confidence": 0.9}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            with patch("backend.output.result_writer._output_dir", out_dir):
                result_writer.save_result(dict(sample))
                result_writer.save_result(dict(sample))

            files = list(out_dir.glob("*.jsonl"))
            self.assertEqual(len(files), 1)

            lines = files[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertIn("timestamp", first)
            self.assertIn("timestamp", second)
            self.assertEqual(first["status"], "OK")


if __name__ == "__main__":
    unittest.main()

