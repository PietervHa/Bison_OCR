"""
Camera rotation and counter reset work directly from the HMI: maintenance
mode (and its password) has been removed.

Run with:  python -m unittest tests.test_web_controls
"""
import unittest

try:
    import httpx  # noqa: F401  (FastAPI's TestClient needs it)
    from fastapi.testclient import TestClient
    from frontend.web import create_app
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False

from backend.core.state import AppState


class _Cam:
    def get_frame(self):
        return None

    def is_connected(self):
        return False


@unittest.skipUnless(HAVE_CLIENT, "pip install httpx to run the web tests")
class WebControlsTests(unittest.TestCase):
    def setUp(self):
        self.state = AppState()
        self.client = TestClient(create_app(_Cam(), self.state))

    def test_reset_counters_needs_no_login(self):
        self.state.increment_counter("OK")
        self.assertEqual(self.client.post("/reset_counters").status_code, 200)
        self.assertEqual(self.state.get_snapshot()["counters"], {"ok": 0, "nok": 0, "total": 0})

    def test_rotate_camera_needs_no_login(self):
        before = self.state.get_camera_rotation()
        self.assertEqual(self.client.post("/camera_rotation", json={}).status_code, 200)
        self.assertNotEqual(self.state.get_camera_rotation(), before)

    def test_maintenance_is_gone(self):
        self.assertNotIn("maintenance_mode", self.client.get("/status").json())
        self.assertEqual(self.client.post("/maintenance_mode", json={}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
