"""
Tests for camera.device_name resolution (backend/core/camera.py).

Guards against the 2026-09-03 08:21 incident in vision.log: a reconnect ran
the pygrabber lookup on a thread without COM initialised
("CoInitialize is niet aangeroepen"), the lookup failed, the code fell back
to camera.index=0, and a 1280x720 device (not the 744x480 DFK) was opened.

pygrabber/comtypes are Windows-only, so they are replaced by fakes that
enforce the same rule real COM does: enumeration fails with
CO_E_NOTINITIALIZED on any thread that hasn't called CoInitialize.

Run with:  python -m unittest tests.test_camera_device_lookup
"""
import sys
import threading
import types
import unittest
from unittest import mock

from backend.core import camera

CO_E_NOTINITIALIZED = -2147221008
CAM_CFG = {"device_name": "DFK 22BUC03", "index": 0}
DEVICES = ["Integrated Webcam", "DFK 22BUC03"]


def _fake_com_modules(devices=DEVICES, enumerate_error=None):
    com_state = threading.local()
    calls = {"init_threads": set(), "uninit": 0}

    comtypes = types.ModuleType("comtypes")

    def co_initialize():
        com_state.ok = True
        calls["init_threads"].add(threading.get_ident())

    def co_uninitialize():
        com_state.ok = False
        calls["uninit"] += 1

    comtypes.CoInitialize = co_initialize
    comtypes.CoUninitialize = co_uninitialize

    class FilterGraph:
        def __init__(self):
            if not getattr(com_state, "ok", False):
                raise OSError(CO_E_NOTINITIALIZED, "CoInitialize is niet aangeroepen")

        def get_input_devices(self):
            if enumerate_error:
                raise enumerate_error
            return list(devices)

    pygrabber = types.ModuleType("pygrabber")
    dshow_graph = types.ModuleType("pygrabber.dshow_graph")
    dshow_graph.FilterGraph = FilterGraph
    pygrabber.dshow_graph = dshow_graph
    modules = {"comtypes": comtypes, "pygrabber": pygrabber, "pygrabber.dshow_graph": dshow_graph}
    return modules, calls


def _resolve_on_new_thread(cam_cfg):
    """Call _resolve_camera_index from a fresh thread, like _spawn_worker does on reconnect."""
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("idx", camera._resolve_camera_index(cam_cfg)))
    t.start()
    t.join(10)
    return out["idx"]


class CameraDeviceLookupTests(unittest.TestCase):
    def setUp(self):
        camera._device_name_lookup_warned = False
        camera._device_enum_failing = False
        patcher = mock.patch.object(camera, "sys", types.SimpleNamespace(platform="win32"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _with_modules(self, modules):
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lookup_works_on_a_reconnect_thread_without_com(self):
        modules, calls = _fake_com_modules()
        self._with_modules(modules)
        self.assertEqual(_resolve_on_new_thread(CAM_CFG), 1)
        self.assertEqual(len(calls["init_threads"]), 1)
        self.assertEqual(calls["uninit"], 1, "CoInitialize must be balanced by CoUninitialize")

    def test_enumeration_failure_opens_nothing_instead_of_index_0(self):
        modules, _ = _fake_com_modules(enumerate_error=OSError(CO_E_NOTINITIALIZED, "boom"))
        self._with_modules(modules)
        with self.assertLogs(camera.log, level="ERROR"):
            self.assertIsNone(_resolve_on_new_thread(CAM_CFG))

    def test_repeated_enumeration_failures_log_error_once(self):
        modules, _ = _fake_com_modules(enumerate_error=OSError(CO_E_NOTINITIALIZED, "boom"))
        self._with_modules(modules)
        with self.assertLogs(camera.log, level="DEBUG") as logs:
            for _ in range(3):
                self.assertIsNone(camera._resolve_camera_index(CAM_CFG))
        errors = [r for r in logs.records if r.levelname == "ERROR"]
        self.assertEqual(len(errors), 1)

    def test_device_not_connected_returns_none(self):
        modules, _ = _fake_com_modules(devices=["Integrated Webcam"])
        self._with_modules(modules)
        self.assertIsNone(camera._resolve_camera_index(CAM_CFG))

    def test_missing_pygrabber_falls_back_to_index(self):
        self._with_modules({"pygrabber": None, "pygrabber.dshow_graph": None})
        with self.assertLogs(camera.log, level="ERROR"):
            self.assertEqual(camera._resolve_camera_index(CAM_CFG), 0)

    def test_no_device_name_uses_index(self):
        self.assertEqual(camera._resolve_camera_index({"device_name": "", "index": 2}), 2)


if __name__ == "__main__":
    unittest.main()
