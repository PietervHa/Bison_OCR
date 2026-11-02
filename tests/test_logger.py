import logging
import logging.config
import unittest

from backend.utils.logger import UVICORN_LOG_CONFIG, _POLL_FILTER


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class PollRequestDemotionTests(unittest.TestCase):
    def setUp(self):
        self.access = logging.getLogger("uvicorn.access")
        self.access.addFilter(_POLL_FILTER)
        # Same call uvicorn.run(log_config=...) makes; it must keep the filter.
        logging.config.dictConfig(UVICORN_LOG_CONFIG)
        self.handler = _Collect()
        self.access.addHandler(self.handler)
        self.access.propagate = False

    def tearDown(self):
        self.access.removeHandler(self.handler)
        self.access.removeFilter(_POLL_FILTER)
        self.access.propagate = True

    def _request(self, method, path, status=200):
        # Matches uvicorn's access-log call signature.
        self.access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", method, path, "1.1", status)

    def test_poll_requests_do_not_reach_info_handlers(self):
        self._request("GET", "/result")
        self._request("GET", "/result?t=123")
        self._request("GET", "/status")
        self._request("GET", "/camera_status")
        self.assertEqual(self.handler.records, [])

    def test_other_requests_stay_at_info(self):
        self._request("POST", "/reset_counters")
        self._request("GET", "/video_feed?t=1")
        self._request("GET", "/test_images")
        self.assertEqual([r.levelno for r in self.handler.records], [logging.INFO] * 3)


if __name__ == "__main__":
    unittest.main()
