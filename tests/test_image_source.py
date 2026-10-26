"""
Test-image mode (camera.source: "images"): ImageFolderSource behaves like the
camera, steps through the folder with wrap-around, picks up new images, and
the HMI routes step and start a read.

Run with:  python -m unittest tests.test_image_source
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.core.image_source import ImageFolderSource
from backend.core.state import AppState

try:
    import httpx  # noqa: F401  (FastAPI's TestClient needs it)
    from fastapi.testclient import TestClient
    from frontend.web import create_app
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False


def _write(folder, name, value, shape=(40, 60, 3)):
    cv2.imwrite(str(Path(folder) / name), np.full(shape, value, np.uint8))


class ImageFolderSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        for name, v in (("b.png", 20), ("a.jpg", 10), ("c.bmp", 30)):
            _write(self.dir, name, v)
        Path(self.dir, "notes.txt").write_text("not an image")

    def tearDown(self):
        self.tmp.cleanup()

    def test_steps_through_images_in_name_order_and_wraps(self):
        src = ImageFolderSource(self.dir)
        self.assertEqual(src.status()["name"], "a.jpg")
        self.assertEqual(src.status()["count"], 3)              # notes.txt ignored
        self.assertEqual(src.step(+1)["name"], "b.png")
        self.assertEqual(src.step(+1)["name"], "c.bmp")
        self.assertEqual(src.step(+1)["name"], "a.jpg")         # wraps forward
        self.assertEqual(src.step(-1)["name"], "c.bmp")         # wraps backward
        self.assertTrue(src.is_connected())
        self.assertEqual(int(src.get_frame()[0, 0, 0]), 30)

    def test_new_images_are_picked_up_and_position_is_kept(self):
        src = ImageFolderSource(self.dir)
        src.step(+1)                                            # b.png
        _write(self.dir, "aa.png", 15)                          # sorts before b.png
        st = src.step(0)
        self.assertEqual((st["name"], st["count"]), ("b.png", 4))

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as empty:
            src = ImageFolderSource(empty)
            self.assertFalse(src.is_connected())
            self.assertIsNone(src.get_frame())
            self.assertEqual(src.step(+1)["count"], 0)

    def test_hmi_rotation_applies(self):
        state = AppState()
        src = ImageFolderSource(self.dir, app_state=state)
        self.assertEqual(src.get_frame().shape[:2], (40, 60))
        state.rotate_camera()
        self.assertEqual(src.get_frame().shape[:2], (60, 40))


class _LiveCam:
    def get_frame(self):
        return None

    def is_connected(self):
        return False


@unittest.skipUnless(HAVE_CLIENT, "pip install httpx to run the web tests")
class TestImageRoutesTests(unittest.TestCase):
    def test_arrow_key_routes_step_and_start_a_read(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "1.png", 1)
            _write(d, "2.png", 2)
            started = []
            client = TestClient(create_app(ImageFolderSource(d), AppState(), start_cycle=lambda: started.append(1) or True))
            self.assertEqual(client.get("/test_images").json()["name"], "1.png")
            r = client.post("/test_images/next").json()
            self.assertEqual((r["name"], r["index"], r["read_started"]), ("2.png", 2, True))
            self.assertEqual(client.post("/test_images/prev").json()["name"], "1.png")
            self.assertEqual(len(started), 2)

    def test_live_camera_has_no_test_images(self):
        client = TestClient(create_app(_LiveCam(), AppState()))
        self.assertEqual(client.get("/test_images").json(), {"source": "live"})
        self.assertEqual(client.post("/test_images/next").status_code, 409)


if __name__ == "__main__":
    unittest.main()
