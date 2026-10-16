"""
Tests for PaddleOCR engine-failure handling and the OCR output filter.

Guards against:
  - a dead / hung / failing PaddleOCR worker being reported as "no text
    detected" instead of an engine error;
  - a dead worker never being restarted;
  - startup failures surfacing only as a bare queue.Empty;
  - CJK/noise characters from the multilingual recognizer reaching the PLC.

The worker tests spawn real subprocesses running the fake loops in
tests/_paddle_worker_fakes.py (no PaddleOCR install needed); they take a
few seconds. Run with:  python -m unittest tests.test_paddle_worker
"""
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from backend.detection.ocr import paddle_worker
from backend.detection.ocr.paddle_worker import PaddleOCRWorker, PaddleWorkerError

try:  # works whether discovered as tests.test_x or as a top-level module
    from tests import _paddle_worker_fakes as fakes
except ImportError:  # pragma: no cover
    import _paddle_worker_fakes as fakes

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def _worker(testcase, loop, lang="en", **kwargs):
    """Build a PaddleOCRWorker whose subprocess (and every restart of it)
    runs `loop` instead of the real PaddleOCR loop."""
    kwargs.setdefault("ready_timeout", 30.0)
    kwargs.setdefault("inference_timeout", 5.0)
    kwargs.setdefault("restart_cooldown_s", 0.0)
    patcher = mock.patch.object(paddle_worker, "_worker_loop", loop)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    worker = PaddleOCRWorker(lang=lang, **kwargs)
    testcase.addCleanup(worker._kill)
    return worker


class PaddleWorkerSupervisionTests(unittest.TestCase):
    def _make(self, loop, **kwargs):
        return _worker(self, loop, **kwargs)

    def test_happy_path_returns_candidates(self):
        w = self._make(fakes.loop_ok)
        self.assertEqual(w.ocr(FRAME), [{"text": "4113 08:25", "confidence": 0.99}])

    def test_init_error_is_reported_with_the_workers_traceback(self):
        with self.assertRaises(PaddleWorkerError) as ctx:
            _worker(self, fakes.loop_init_error)
        self.assertIn("unknown model", str(ctx.exception))

    def test_worker_exiting_before_ready_is_reported_with_exit_code(self):
        with self.assertRaises(PaddleWorkerError) as ctx:
            _worker(self, fakes.loop_exit_without_ready, ready_timeout=5.0)
        self.assertIn("exit", str(ctx.exception))

    def test_predict_error_raises_instead_of_returning_empty(self):
        w = self._make(fakes.loop_predict_error)
        with self.assertRaises(PaddleWorkerError) as ctx:
            w.ocr(FRAME)
        self.assertIn("ConvertPirAttribute2RuntimeAttribute", str(ctx.exception))
        self.assertTrue(w.is_alive(), "a per-call error must not kill a healthy worker")

    def test_crashed_worker_is_detected_and_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "crashed")
            w = self._make(fakes.loop_crash_first_job_only, lang=marker, inference_timeout=30.0)
            start = time.monotonic()
            with self.assertRaises(PaddleWorkerError) as ctx:
                w.ocr(FRAME)
            self.assertIn("died", str(ctx.exception))
            self.assertLess(time.monotonic() - start, 5.0, "crash should be noticed without waiting out the timeout")
            # Next cycle: restarted transparently and working again.
            self.assertEqual(w.ocr(FRAME)[0]["text"], "after restart")
            self.assertTrue(w.is_alive())

    def test_process_killed_between_calls_is_restarted(self):
        w = self._make(fakes.loop_ok)
        w._process.terminate()
        w._process.join(5)
        self.assertEqual(w.ocr(FRAME)[0]["text"], "4113 08:25")

    def test_hung_worker_times_out_and_is_replaced(self):
        w = self._make(fakes.loop_hang, inference_timeout=1.0)
        start = time.monotonic()
        with self.assertRaises(PaddleWorkerError) as ctx:
            w.ocr(FRAME)
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertIn("did not answer", str(ctx.exception))
        self.assertFalse(w.is_alive(), "hung worker should have been terminated")

    def test_restart_cooldown_fails_fast(self):
        w = self._make(fakes.loop_ok, restart_cooldown_s=60.0)
        # Worker dies, and a restart was "just" attempted -> must fail fast.
        w._process.terminate()
        w._process.join(5)
        w._last_start_attempt = time.monotonic()  # pretend we just tried
        start = time.monotonic()
        with self.assertRaises(PaddleWorkerError) as ctx:
            w.ocr(FRAME)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertIn("next restart attempt", str(ctx.exception))


class _FakeWorker:
    """Stands in for PaddleOCRWorker inside PaddleOCR (no subprocess)."""
    responses = []

    def __init__(self, **kwargs):
        self.calls = 0

    def ocr(self, frame, timeout=None):
        self.calls += 1
        resp = _FakeWorker.responses[min(self.calls - 1, len(_FakeWorker.responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return [dict(c) for c in resp]


class PaddleOCRResultTests(unittest.TestCase):
    def _ocr(self, *responses):
        from backend.detection.ocr import paddle_ocr
        _FakeWorker.responses = list(responses)
        with mock.patch.object(paddle_ocr, "PaddleOCRWorker", _FakeWorker):
            engine = paddle_ocr.PaddleOCR(app_state=None)
        engine.dynamic_roi_debug_dir = None
        engine.derib_enabled = False
        engine.orientation_retry_enabled = True
        engine.orientation_retry_require_low_confidence = True
        engine.orientation_retry_confidence_threshold = 0.8
        # These tests cover the raw-read path; validation has its own tests.
        engine.validator = None
        engine.consensus_variants = [None]
        return engine

    def _blank_frame(self):
        return np.full((480, 744, 3), 190, dtype=np.uint8)

    def test_engine_failure_is_an_error_not_an_empty_read(self):
        engine = self._ocr(PaddleWorkerError("worker died"))
        result = engine.read(self._blank_frame())
        self.assertIn("worker died", result.get("error", ""))
        self.assertEqual(result["detections"], [])
        self.assertEqual(engine._paddle.calls, 1, "must not run the orientation retry after an engine failure")

    def test_retry_failure_after_empty_first_pass_is_an_error(self):
        engine = self._ocr([], PaddleWorkerError("died on retry"))
        result = engine.read(self._blank_frame())
        self.assertIn("died on retry", result.get("error", ""))

    def test_retry_failure_keeps_a_usable_first_pass(self):
        engine = self._ocr([{"text": "4113 08:25", "confidence": 0.5}], PaddleWorkerError("died on retry"))
        result = engine.read(self._blank_frame())
        self.assertNotIn("error", result)
        self.assertEqual(result["text"], "4113 08:25")

    def test_successful_read_has_no_error(self):
        engine = self._ocr([{"text": "4113 08:25", "confidence": 0.99}])
        result = engine.read(self._blank_frame())
        self.assertNotIn("error", result)
        self.assertEqual(result["text"], "4113 08:25")

    def test_output_filter_drops_noise_tokens_and_marks_bad_characters(self):
        engine = self._ocr([
            {"text": "心", "confidence": 0.7},
            {"text": "4113 08:25", "confidence": 0.99},
            {"text": "BB 0门/26", "confidence": 0.95},
        ])
        result = engine.read(self._blank_frame())
        self.assertEqual(result["text"], "4113 08:25 BB 0?/26")
        self.assertTrue(result["text"].isascii())

    def test_output_filter_can_be_disabled(self):
        engine = self._ocr([{"text": "心", "confidence": 0.7}])
        engine._allowed_chars = None
        self.assertEqual(engine.read(self._blank_frame())["text"], "心")


class VisionNormalizationTests(unittest.TestCase):
    def test_ocread_error_becomes_ocr_engine_error(self):
        try:
            from backend.core import vision
        except Exception as exc:  # torch/transformers not installed in this environment
            self.skipTest(f"backend.core.vision not importable here: {exc}")
        normalized = vision._normalize_result(
            {"detections": [], "text": "", "mode": "ocread", "error": "worker died"}
        )
        self.assertEqual(normalized["status"], "NOK")
        self.assertEqual(normalized["failure_reason"], "ocr_engine_error")


if __name__ == "__main__":
    unittest.main()
