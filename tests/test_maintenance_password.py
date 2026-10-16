"""
Tests for the maintenance-mode password (MAINTENANCE_PASSWORD).

Rules protected here: maintenance mode is locked when no password is
configured, refused (after a delay) on a wrong password, and unlocked -
with a session cookie that the protected routes accept - on the right one.

Needs httpx for FastAPI's TestClient; skipped when it is not installed.
Run with:  python -m unittest tests.test_maintenance_password
"""
import importlib.util
import os
import unittest
from unittest import mock

if importlib.util.find_spec("httpx") is not None:
    from fastapi.testclient import TestClient
else:  # pragma: no cover - depends on the environment
    TestClient = None

from backend.core.state import AppState


class _Camera:
    def get_frame(self):
        return None

    def is_connected(self):
        return False


def _env_without_password():
    return {k: v for k, v in os.environ.items() if k != "MAINTENANCE_PASSWORD"}


@unittest.skipIf(TestClient is None, "httpx is not installed")
class MaintenancePasswordTests(unittest.TestCase):
    def setUp(self):
        from frontend import web
        self.web = web
        self.state = AppState()
        self.client = TestClient(web.create_app(_Camera(), self.state))
        # Never actually wait in tests; that the delay runs is asserted.
        delay_patch = mock.patch.object(web, "_wrong_password_delay")
        self.delay = delay_patch.start()
        self.addCleanup(delay_patch.stop)

    def _login(self, password):
        return self.client.post("/maintenance_mode", json={"maintenance_mode": True, "password": password})

    def _in_maintenance(self):
        return self.client.get("/status").json()["maintenance_mode"]

    def test_locked_when_password_is_not_set(self):
        with mock.patch.dict(os.environ, _env_without_password(), clear=True):
            for attempt in ("", "anything"):
                res = self._login(attempt)
                self.assertEqual(res.status_code, 403)
                self.assertEqual(res.json()["error"], "not_configured")
        self.assertFalse(self.state.get_maintenance_mode())
        self.assertFalse(self._in_maintenance())
        self.assertEqual(self.client.post("/reset_counters").status_code, 403)

    def test_locked_when_password_is_empty(self):
        with mock.patch.dict(os.environ, {"MAINTENANCE_PASSWORD": ""}):
            res = self._login("")
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["error"], "not_configured")
        self.assertFalse(self.state.get_maintenance_mode())

    def test_wrong_password_is_refused_after_a_delay(self):
        with mock.patch.dict(os.environ, {"MAINTENANCE_PASSWORD": "s3cret"}):
            res = self._login("wrong")
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["error"], "invalid_password")
        self.assertNotIn("maintenance_session", res.cookies)
        self.delay.assert_called_once()
        self.assertGreaterEqual(self.web._WRONG_PASSWORD_DELAY_S, 1.0)
        self.assertFalse(self.state.get_maintenance_mode())
        self.assertFalse(self._in_maintenance())
        self.assertEqual(self.client.post("/reset_counters").status_code, 403)

    def test_right_password_unlocks_with_a_session_cookie(self):
        with mock.patch.dict(os.environ, {"MAINTENANCE_PASSWORD": "s3cret"}):
            res = self._login("s3cret")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"maintenance_mode": True})
        self.assertIn("maintenance_session", res.cookies)
        self.delay.assert_not_called()
        self.assertTrue(self._in_maintenance())
        self.assertEqual(self.client.post("/reset_counters").status_code, 200)

        # A client without the session cookie still has no access.
        other = TestClient(self.client.app)
        self.assertFalse(other.get("/status").json()["maintenance_mode"])
        self.assertEqual(other.post("/reset_counters").status_code, 403)

        # Leaving maintenance mode locks it again.
        res = self.client.post("/maintenance_mode", json={"maintenance_mode": False})
        self.assertEqual(res.json(), {"maintenance_mode": False})
        self.assertFalse(self._in_maintenance())


if __name__ == "__main__":
    unittest.main()
