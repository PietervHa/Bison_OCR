"""
Tests for the OCRead response sent to the PLC by TCPTriggerServer.

Run with:  python -m unittest tests.test_tcp_trigger_ocread
"""
import unittest

import numpy as np

from backend.core.tcp_trigger_server import TCPTriggerServer


class _Camera:
    def get_frame(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _server(result, recorded):
    def run_vision(frame, callback=None):
        callback(dict(result))

    srv = TCPTriggerServer(
        camera=_Camera(),
        run_vision_fn=run_vision,
        process_result_fn=lambda res, t: recorded.append(res),
    )
    srv.response_ok, srv.response_nok = b"OK\n", b"NOK\n"
    srv.max_text_length = 64
    return srv


def _ocread(text, status="OK"):
    return {
        "mode": "ocread",
        "status": status,
        "text": text,
        "detections": [{"text": text, "confidence": 0.99}],
    }


class OCReadResponseTests(unittest.TestCase):
    def test_successful_read_sends_text_line(self):
        recorded = []
        response = _server(_ocread("4113 08:25 BB 05/26"), recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"4113 08:25 BB 05/26\n")
        self.assertEqual(recorded[0]["status"], "OK")

    def test_text_never_contains_an_embedded_line_break(self):
        recorded = []
        response = _server(_ocread("4113 08:25\nBB 05/26\r"), recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"4113 08:25 BB 05/26\n")
        self.assertEqual(response.count(b"\n"), 1)

    def test_over_long_read_is_sent_and_recorded_as_nok(self):
        recorded = []
        response = _server(_ocread("X" * 65), recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"NOK\n")
        self.assertEqual(recorded[0]["status"], "NOK")
        self.assertEqual(recorded[0]["failure_reason"], "text_too_long")

    def test_read_at_the_limit_is_still_sent(self):
        recorded = []
        response = _server(_ocread("X" * 64), recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"X" * 64 + b"\n")

    def test_nok_result_still_sends_nok(self):
        recorded = []
        result = _ocread("", status="NOK")
        result["failure_reason"] = "no_text_detected"
        response = _server(result, recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"NOK\n")

    def test_result_without_ocread_mode_gets_plain_ok_nok(self):
        recorded = []
        result = {"status": "OK", "text": "X" * 200, "detections": []}
        response = _server(result, recorded)._trigger_vision_and_wait()
        self.assertEqual(response, b"OK\n")


if __name__ == "__main__":
    unittest.main()
