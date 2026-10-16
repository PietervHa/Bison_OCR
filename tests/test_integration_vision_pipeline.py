import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core import vision
from backend.output import result_writer


class _FakeAppState:
    def __init__(self, mode="object_detection", threshold=0.8):
        self._mode = mode
        self._threshold = threshold
        self.classifier_updates = []

    def get_vision_mode(self):
        return self._mode

    def get_threshold(self):
        return self._threshold

    def set_classifier_loaded(self, value):
        self.classifier_updates.append(value)


class _FakeEngine:
    def __init__(self, response):
        self._response = response

    def evaluate(self, _frame):
        return dict(self._response)


class VisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._old_app_state = vision._app_state
        self._old_engine = vision._inspection_engine
        self._old_cached_mode = vision._cached_vision_mode

    def tearDown(self):
        vision._app_state = self._old_app_state
        vision._inspection_engine = self._old_engine
        vision._cached_vision_mode = self._old_cached_mode

    def test_run_vision_object_detection_applies_threshold_normalization(self):
        vision._app_state = _FakeAppState(mode="object_detection", threshold=0.8)
        vision._inspection_engine = _FakeEngine(
            {
                "status": "OK",
                "detections": [{"label": "cap", "confidence": 0.92}],
                "processing_time_ms": 2.1,
            }
        )

        result = vision.run_vision(frame=object())

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["mode"], "object_detection")
        self.assertAlmostEqual(result["confidence"], 0.92)

    def test_run_vision_forces_nok_when_confidence_below_threshold(self):
        vision._app_state = _FakeAppState(mode="object_detection", threshold=0.8)
        vision._inspection_engine = _FakeEngine(
            {
                "status": "OK",
                "detections": [{"label": "cap", "confidence": 0.2}],
                "processing_time_ms": 1.0,
            }
        )

        result = vision.run_vision(frame=object())
        self.assertEqual(result["status"], "NOK")

    def test_run_vision_returns_nok_for_missing_frame(self):
        vision._app_state = _FakeAppState(mode="object_detection", threshold=0.8)
        vision._inspection_engine = _FakeEngine({"status": "OK"})

        result = vision.run_vision(frame=None)

        self.assertEqual(result["status"], "NOK")
        self.assertEqual(result["error"], "no_frame")
        self.assertEqual(result["detections"], [])

    def test_run_vision_callback_path_returns_normalized_payload(self):
        vision._app_state = _FakeAppState(mode="object_detection", threshold=0.1)
        vision._inspection_engine = _FakeEngine(
            {
                "label": "non-defective",
                "confidence": 0.9,
                "processing_time_ms": 1.0,
            }
        )

        event = threading.Event()
        captured = {}

        def callback(result):
            captured.update(result)
            event.set()

        vision.run_vision(frame=object(), callback=callback)
        self.assertTrue(event.wait(2.0), "vision callback did not complete in time")

        self.assertEqual(captured["status"], "OK")
        self.assertEqual(captured["mode"], "object_detection")
        self.assertEqual(len(captured["detections"]), 1)
        self.assertEqual(captured["detections"][0]["label"], "non-defective")

    def test_run_vision_callback_path_returns_nok_for_missing_frame(self):
        vision._app_state = _FakeAppState(mode="object_detection", threshold=0.1)
        vision._inspection_engine = _FakeEngine({"status": "OK"})

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

