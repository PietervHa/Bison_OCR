"""
Test-image mode (camera.source: "images"): ImageFolderSource behaves like the
camera, steps through the folder with wrap-around, picks up new images, and
the HMI routes step and start a read.

Run with:  python -m unittest tests.test_image_source
"""
import datetime
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend.core import vision
from backend.core.image_source import ImageFolderSource
from backend.core.state import AppState

try:
    importlib.import_module("httpx")  # FastAPI's TestClient needs it
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

    def test_stays_connected_while_stepping(self):
        # Simulate a slow load: is_connected() polled mid-step (as /camera_status
        # does) must not see the source as disconnected.
        src = ImageFolderSource(self.dir)
        loading, release = threading.Event(), threading.Event()
        real_read = ImageFolderSource._read

        def slow_read(path):
            loading.set()
            release.wait(5)
            return real_read(path)

        with mock.patch.object(ImageFolderSource, "_read", staticmethod(slow_read)):
            stepper = threading.Thread(target=src.step, args=(+1,))
            stepper.start()
            self.assertTrue(loading.wait(5))
            self.assertTrue(src.is_connected())
            release.set()
            stepper.join(5)
        self.assertTrue(src.is_connected())
        self.assertEqual(int(src.get_frame()[0, 0, 0]), 20)          # b.png

    def test_unreadable_file_gives_no_frame_and_stays_connected(self):
        Path(self.dir, "b2.png").write_bytes(b"not a png")             # sorts after b.png
        src = ImageFolderSource(self.dir)
        src.step(+1)                                                  # b.png
        st = src.step(+1)                                             # b2.png
        self.assertEqual((st["name"], st["unreadable"]), ("b2.png", "b2.png"))
        self.assertTrue(src.is_connected())
        self.assertIsNone(src.get_frame())                            # not b.png under b2's name
        self.assertIsNone(src.frame_capture_time())
        self.assertIsNone(src.step(+1)["unreadable"])                 # c.bmp
        self.assertEqual(int(src.get_frame()[0, 0, 0]), 30)

    def test_cycle_on_unreadable_file_is_nok_and_next_file_reads_with_its_time(self):
        # One cycle as main.start_vision_cycle runs it: frame + capture time
        # from the source, then run_vision. The OCR engine is a stub that
        # records what it was given.
        with tempfile.TemporaryDirectory() as d:
            _write(d, "UId_0_Time_1-10-2026_6-32-50_738.png", 10)
            Path(d, "UId_0_Time_1-10-2026_6-33-10_100.png").write_bytes(b"not a png")
            _write(d, "UId_0_Time_1-10-2026_6-33-30_200.png", 30)
            calls = []

            def fake_read(frame, capture_time=None, **_):
                calls.append((int(frame[0, 0, 0]), capture_time))
                return {"status": "OK", "text": "CODE", "detections": [{"text": "CODE"}]}

            def cycle(src):
                return vision.run_vision(src.get_frame(), capture_time=vision.capture_time_of(src))

            src = ImageFolderSource(d)
            with mock.patch.object(vision, "ocr_instance", mock.Mock(read=fake_read)):
                self.assertEqual(src.step(+1)["unreadable"], "UId_0_Time_1-10-2026_6-33-10_100.png")
                r = cycle(src)
                self.assertEqual((r["status"], r["text"]), ("NOK", ""))
                self.assertEqual(calls, [])                           # the old image was not read

                src.step(+1)
                r = cycle(src)
                self.assertEqual((r["status"], r["text"]), ("OK", "CODE"))
                self.assertEqual(calls, [(30, datetime.datetime(2026, 10, 1, 6, 33, 30))])

    def test_only_unreadable_files_is_not_connected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "bad.png").write_bytes(b"")
            src = ImageFolderSource(d)
            self.assertFalse(src.is_connected())
            self.assertEqual(src.status()["unreadable"], "bad.png")

    def test_unreadable_first_file_is_connected_when_others_are_readable(self):
        Path(self.dir, "0.png").write_bytes(b"not a png")             # sorts first
        src = ImageFolderSource(self.dir)
        self.assertEqual(src.status()["unreadable"], "0.png")
        self.assertTrue(src.is_connected())

    def test_reads_paths_with_special_characters(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d, "tëst_ß")
            folder.mkdir()
            _, buf = cv2.imencode(".png", np.full((40, 60, 3), 77, np.uint8))
            buf.tofile(str(folder / "beeld_é.png"))
            src = ImageFolderSource(folder)
            self.assertTrue(src.is_connected())
            self.assertEqual(int(src.get_frame()[0, 0, 0]), 77)

    def test_hmi_rotation_applies(self):
        state = AppState()
        src = ImageFolderSource(self.dir, app_state=state)
        self.assertEqual(src.get_frame().shape[:2], (40, 60))
        state.rotate_camera()
        self.assertEqual(src.get_frame().shape[:2], (60, 40))


class ImageScalingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_wider_image_is_scaled_to_max_width_keeping_aspect_ratio(self):
        _write(self.dir, "big.png", 50, shape=(876, 1596, 3))
        frame = ImageFolderSource(self.dir, max_width=744).get_frame()
        self.assertEqual(frame.shape[:2], (round(876 * 744 / 1596), 744))   # 408 x 744

    def test_narrower_image_keeps_its_size(self):
        _write(self.dir, "small.png", 50, shape=(300, 500, 3))
        self.assertEqual(ImageFolderSource(self.dir, max_width=744).get_frame().shape[:2], (300, 500))

    def test_max_width_zero_keeps_original_size(self):
        _write(self.dir, "big.png", 50, shape=(876, 1596, 3))
        self.assertEqual(ImageFolderSource(self.dir, max_width=0).get_frame().shape[:2], (876, 1596))

    def test_hmi_rotation_applies_after_scaling(self):
        _write(self.dir, "big.png", 50, shape=(876, 1596, 3))
        state = AppState()
        src = ImageFolderSource(self.dir, app_state=state, max_width=744)
        state.rotate_camera()
        self.assertEqual(src.get_frame().shape[:2], (744, 408))


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
