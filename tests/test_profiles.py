"""
Product profiles (backend/core/profiles.py, config/profiles/*.yaml), the ink
channel (backend/detection/ocr/ink_channel.py), the runtime switch
(backend/core/vision.py) and the HMI routes (frontend/web.py).

The rule being protected: the standard profile reads exactly as the config
file does (the 514 test images: 502 correct, 0 WRONG), and the blue
profile's settings can't leak into it.

Run with:  python -m unittest tests.test_profiles
"""
import importlib
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend.core import profiles
from backend.core.config_loader import cfg
from backend.detection.ocr.ink_channel import InkChannel

FIX = Path(__file__).resolve().parent / "fixtures"

try:
    importlib.import_module("httpx")  # FastAPI's TestClient needs it
    from fastapi.testclient import TestClient
    from frontend.web import create_app
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False


def _write(directory, name, text):
    Path(directory, f"{name}.yaml").write_text(text, encoding="utf-8")


class ProfileFilesTests(unittest.TestCase):
    def test_standard_profile_is_the_config_file_as_it_is(self):
        self.assertEqual(profiles.config_for("standard"), cfg)

    def test_standard_without_its_file_is_the_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(profiles.config_for("standard", directory=d), cfg)
            self.assertEqual(profiles.names(d), ["standard"])

    def test_names_standard_first(self):
        names = profiles.names()
        self.assertEqual(names[0], "standard")
        self.assertIn("blue", names)

    def test_only_ocr_can_be_changed(self):
        # A profile must not change the PLC trigger, camera, web, output ...
        with tempfile.TemporaryDirectory() as d:
            _write(d, "bad", "label: x\ntrigger:\n  port: 6000\nocr: {}\n")
            with self.assertRaises(profiles.ProfileError):
                profiles.config_for("bad", directory=d)
            self.assertIn("invalid", profiles.describe("bad", d)["label"])

    def test_invalid_or_missing_names_are_refused(self):
        for name in ("../default", "a b", "", None, "blue.yaml", "does-not-exist"):
            with self.subTest(name=name), self.assertRaises(profiles.ProfileError):
                profiles.load(name)

    def test_mappings_merge_and_lists_replace(self):
        base = {"ocr": {"a": 1, "dynamic_roi": {"x": 1, "y": 2}, "read_validation": {"templates": [[1], [2]]}},
                "trigger": {"port": 5001}}
        with tempfile.TemporaryDirectory() as d:
            _write(d, "p", "ocr:\n  dynamic_roi: {y: 3}\n  read_validation: {templates: [[9]]}\n")
            out = profiles.config_for("p", base=base, directory=d)
        self.assertEqual(out["ocr"], {"a": 1, "dynamic_roi": {"x": 1, "y": 3}, "read_validation": {"templates": [[9]]}})
        self.assertEqual(out["trigger"], {"port": 5001})
        self.assertEqual(base["ocr"]["dynamic_roi"], {"x": 1, "y": 2}, "the base config must not change")

    def test_blue_profile(self):
        c = profiles.config_for("blue")
        self.assertEqual(c["ocr"]["ink_channel"]["weights_bgr"], [1.0, 0.0, 0.0])
        self.assertEqual(c["ocr"]["read_validation"]["font_reader"]["model"],
                         "backend/detection/ocr/models/font_reader_blue.npz")
        # Same formats and checks as the config file; only the reading differs.
        for key in ("templates", "value_pattern", "agreement_band", "max_expiry_years", "time_check"):
            self.assertEqual(c["ocr"]["read_validation"].get(key), cfg["ocr"]["read_validation"].get(key), key)
        fr, base_fr = c["ocr"]["read_validation"]["font_reader"], cfg["ocr"]["read_validation"]["font_reader"]
        self.assertEqual(fr["min_digit_confidence"], base_fr["min_digit_confidence"])
        for key in ("camera", "trigger", "web", "output"):
            self.assertEqual(c[key], cfg[key], key)


class InkChannelTests(unittest.TestCase):
    def test_no_setting_is_no_channel(self):
        for c in (None, {}):
            self.assertIsNone(InkChannel.from_config(c))

    def test_invalid_settings_raise(self):
        for c in ("B", {"weights_bgr": [1, 0]}, {"weight": [1, 0, 0]}, {"weights_bgr": [1, 0, 0], "gain": 2}):
            with self.subTest(c=c), self.assertRaises(ValueError):
                InkChannel.from_config(c)

    def test_channel_values(self):
        f = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
        blue = InkChannel([1, 0, 0]).apply(f)
        self.assertTrue(all(np.array_equal(blue[..., k], f[..., 0]) for k in range(3)))
        warm = InkChannel([1, 0, -1], offset=128).apply(f)[..., 0]
        self.assertTrue(np.array_equal(warm, np.clip(f[..., 0].astype(int) - f[..., 2] + 128, 0, 255)))
        # Downstream grayscale gives the channel back exactly.
        self.assertTrue(np.array_equal(cv2.cvtColor(warm[..., None].repeat(3, 2), cv2.COLOR_BGR2GRAY), warm))

    def test_standard_engine_reads_the_frame_itself(self):
        from backend.detection.ocread.engine import OCRead
        frame = np.zeros((8, 8, 3), np.uint8)
        e = OCRead(profiles.config_for("standard"), profile_name="standard")
        self.assertIsNone(e.ink_channel)
        self.assertIs(e.ink_frame(frame), frame)

    def test_locate_blur_off_is_the_same_array(self):
        from backend.detection.ocr.text_locator import _locate_blur
        gray = np.zeros((8, 8), np.uint8)
        self.assertIs(_locate_blur(gray, {}), gray)
        self.assertIs(_locate_blur(gray, {"locate_blur_sigma": 0}), gray)


def _strip(r):
    """A result without what differs from run to run (times) or by name."""
    out = {k: v for k, v in r.items() if k not in ("processing_time_ms", "profile")}
    if "validation" in out:
        reads = [{k: v for k, v in rd.items() if k not in ("timings_ms", "step_ms", "agreement")}
                 for rd in out["validation"]["reads"]]
        for rd in reads:
            rd["formats_tried"] = [{k: v for k, v in a.items() if k != "ms"} for a in rd["formats_tried"]]
        out["validation"] = dict(out["validation"], reads=reads)
    return out


class EngineProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def _engine(self, name=None):
        from backend.detection.ocread.engine import OCRead
        e = OCRead() if name is None else OCRead(profiles.config_for(name), profile_name=name)
        e.dynamic_roi_debug_dir = None
        e.time_budget_ms = 0                          # results, not timing
        return e

    def test_standard_profile_reads_like_the_config_file(self):
        plain, standard = self._engine(), self._engine("standard")
        frames = [cv2.imread(str(p)) for p in sorted((FIX / "font_reader").glob("*.png"))]
        frames += [cv2.imread(str(p)) for p in sorted((FIX / "profiles").glob("*.png"))]
        self.assertGreaterEqual(len(frames), 5)
        for i, frame in enumerate(frames):
            with self.subTest(frame=i):
                a, b = plain.read(frame), standard.read(frame)
                self.assertNotIn("profile", a)
                self.assertEqual(b["profile"], "standard")
                self.assertEqual(_strip(a), _strip(b))

    def test_unavailable_engine_records_its_profile(self):
        from backend.detection.ocread.engine import OCRead
        c = profiles.merge(cfg, {"ocr": {"read_validation": {"font_reader": {"model": "does/not/exist.npz"}}}})
        r = OCRead(c, profile_name="x").read(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual((r["profile"], r["text"]), ("x", ""))
        self.assertIn("error", r)


class BlueBottleTests(unittest.TestCase):
    """Real blue-bottle frames (dynamic_roi debug captures, 2026-11-05):
    unreadable with the standard profile, found and read by the blue
    profile's pipeline (blue channel, locator settings). The pipeline is
    tested with the shared model, which reads these two frames - not with
    font_reader_blue.npz, which is trained on site (see blue.yaml) - so the
    test does not change when that model is (re)trained. Regression tests
    of the pipeline, not an accuracy measurement."""

    FRAMES = {"blue_6303_1045.png": "6303 10:45 BB 05/28", "blue_6303_1044_on_embossing.png": "6303 10:44 BB 05/28"}

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        from backend.detection.ocread.engine import OCRead
        blue = profiles.config_for("blue")
        blue["ocr"]["read_validation"]["font_reader"]["model"] = ""
        cls.engines = {"standard": OCRead(profiles.config_for("standard"), profile_name="standard"),
                       "blue": OCRead(blue, profile_name="blue")}
        for e in cls.engines.values():
            e.dynamic_roi_debug_dir, e.time_budget_ms = None, 0

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_blue_profile_reads_the_blue_bottle(self):
        for name, code in self.FRAMES.items():
            with self.subTest(name=name):
                r = self.engines["blue"].read(cv2.imread(str(FIX / "profiles" / name)))
                self.assertEqual((r["text"], r["profile"]), (code, "blue"))

    def test_standard_profile_does_not_read_it(self):
        for name in self.FRAMES:
            with self.subTest(name=name):
                r = self.engines["standard"].read(cv2.imread(str(FIX / "profiles" / name)))
                self.assertEqual(r["text"], "")

    def test_the_blue_channel_is_what_the_locator_and_reader_see(self):
        frame = cv2.imread(str(FIX / "profiles" / "blue_6303_1045.png"))
        e = self.engines["blue"]
        seen = []
        real = e._prepare

        def spy(f):
            seen.append(f)
            return real(f)
        with mock.patch.object(e, "_prepare", spy):
            e.read(frame)
        self.assertTrue(np.array_equal(seen[0], e.ink_channel.apply(frame)))        # the blue channel ...
        self.assertFalse(np.array_equal(seen[0][..., 0], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))  # ... not grey

    def test_blue_profile_without_its_model_cannot_be_switched_to(self):
        # Until font_reader_blue.npz is trained and copied in, choosing "blue"
        # on the HMI is refused and the active profile stays.
        from backend.core import vision
        model = profiles.PROJECT_ROOT / "backend/detection/ocr/models/font_reader_blue.npz"
        if model.exists():
            self.skipTest("font_reader_blue.npz is installed")
        self.assertIsNone(vision.build_engine("blue").font_reader)
        saved = vision.ocr_instance, vision.active_profile
        try:
            with mock.patch.object(vision, "_remember_profile", lambda name: None):
                with self.assertRaises(profiles.ProfileError):
                    vision.switch_profile("blue")
            self.assertEqual((vision.ocr_instance, vision.active_profile), saved)
        finally:
            vision.ocr_instance, vision.active_profile = saved


class _FakeEngine:
    def __init__(self, name, usable=True):
        self.profile_name, self.font_reader = name, (object() if usable else None)
        self.unavailable_reason = None if usable else "model missing"
        self.thread = None

    def read(self, frame, **_):
        self.thread = threading.current_thread().name
        return {"text": "", "detections": [], "profile": self.profile_name}


class VisionSwitchTests(unittest.TestCase):
    def setUp(self):
        from backend.core import vision
        self.vision = vision
        self.saved = (vision.ocr_instance, vision.active_profile)
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "active_profile_test.txt"
        patches = [mock.patch.object(vision, "_state_file", lambda: self.state),
                   mock.patch.dict(vision.cfg, {"profiles": {"default": "standard", "remember": True}})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        vision.ocr_instance, vision.active_profile = _FakeEngine("standard"), "standard"

    def tearDown(self):
        self.vision.ocr_instance, self.vision.active_profile = self.saved
        self.tmp.cleanup()

    def test_switch_builds_on_the_vision_thread_and_is_remembered(self):
        built = []

        def build(name):
            built.append(threading.current_thread().name)
            return _FakeEngine(name)
        with mock.patch.object(self.vision, "build_engine", build):
            status = self.vision.switch_profile("blue")
        self.assertEqual(status["active"], "blue")
        self.assertEqual(self.vision.ocr_instance.profile_name, "blue")
        self.assertTrue(built[0].startswith("vision_worker"), built)
        self.assertEqual(self.state.read_text().strip(), "blue")
        self.assertEqual(self.vision.startup_profile(), "blue")

    def test_profile_that_cannot_read_is_refused_and_nothing_changes(self):
        before = self.vision.ocr_instance
        with mock.patch.object(self.vision, "build_engine", lambda name: _FakeEngine(name, usable=False)):
            with self.assertRaises(profiles.ProfileError):
                self.vision.switch_profile("blue")
        with self.assertRaises(profiles.ProfileError):
            self.vision.switch_profile("does-not-exist")
        self.assertIs(self.vision.ocr_instance, before)
        self.assertEqual(self.vision.active_profile, "standard")
        self.assertFalse(self.state.exists())

    def test_a_read_queued_behind_a_switch_uses_the_new_profile(self):
        release = threading.Event()

        def slow_build(name):
            release.wait(5)
            return _FakeEngine(name)
        done, results = threading.Event(), []
        with mock.patch.object(self.vision, "build_engine", slow_build):
            # The switch is queued first (as switch_profile does), the read after it.
            switch = self.vision._pool.submit(self.vision._switch, "blue")
            self.vision.run_vision(np.zeros((8, 8, 3), np.uint8),
                                   callback=lambda r: (results.append(r), done.set()))
            release.set()
            switch.result(5)
            self.assertTrue(done.wait(5))
        self.assertEqual(results[0]["profile"], "blue")

    def test_startup_profile(self):
        self.assertEqual(self.vision.startup_profile(), "standard")
        self.state.write_text("does-not-exist\n")
        self.assertEqual(self.vision.startup_profile(), "standard")
        self.state.write_text("blue\n")
        self.assertEqual(self.vision.startup_profile(), "blue")
        with mock.patch.dict(self.vision.cfg, {"profiles": {"default": "standard", "remember": False}}):
            self.assertEqual(self.vision.startup_profile(), "standard")


@unittest.skipUnless(HAVE_CLIENT, "pip install httpx to run the web tests")
class ProfileRoutesTests(unittest.TestCase):
    class _Cam:
        def get_frame(self):
            return None

        def is_connected(self):
            return False

    def _client(self, switch=None):
        from backend.core.state import AppState
        status = {"active": "standard", "profiles": [{"name": "standard", "label": "Standard", "description": ""},
                                                     {"name": "blue", "label": "Blue bottle", "description": ""}]}

        def do_switch(name):
            if name != "blue":
                raise profiles.ProfileError(f"profile {name!r} not found")
            status["active"] = name
            return status
        return TestClient(create_app(self._Cam(), AppState(), profile_status=lambda: status,
                                     switch_profile=switch or do_switch))

    def test_list_and_switch(self):
        client = self._client()
        self.assertEqual(client.get("/profiles").json()["active"], "standard")
        r = client.post("/profile", json={"name": "blue"})
        self.assertEqual((r.status_code, r.json()["active"]), (200, "blue"))
        self.assertEqual(client.get("/profiles").json()["active"], "blue")

    def test_refused_switch_says_why(self):
        r = self._client().post("/profile", json={"name": "green"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("not found", r.json()["error"])

    def test_without_profiles(self):
        from backend.core.state import AppState
        client = TestClient(create_app(self._Cam(), AppState()))
        self.assertEqual(client.get("/profiles").json(), {"active": None, "profiles": []})
        self.assertEqual(client.post("/profile", json={"name": "blue"}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
