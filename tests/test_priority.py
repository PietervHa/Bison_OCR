"""
Tests for backend/utils/priority.py (Windows process / thread priority).

Run with:  python -m unittest tests.test_priority
"""
import sys
import threading
import unittest

from backend.utils.priority import _kernel32, set_process_priority, set_thread_priority


class PriorityTests(unittest.TestCase):
    def test_unknown_names_are_not_set(self):
        self.assertFalse(set_process_priority("realtime"))
        self.assertFalse(set_thread_priority("time_critical"))

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_thread_priority_is_set_on_the_calling_thread(self):
        k32 = _kernel32()
        seen = []

        def worker():
            seen.append(set_thread_priority("highest"))
            seen.append(k32.GetThreadPriority(k32.GetCurrentThread()))

        t = threading.Thread(target=worker)
        t.start(); t.join()
        self.assertEqual(seen, [True, 2])
        self.assertEqual(k32.GetThreadPriority(k32.GetCurrentThread()), 0)   # this thread unchanged

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_process_priority_round_trip(self):
        k32 = _kernel32()
        before = k32.GetPriorityClass(k32.GetCurrentProcess())
        try:
            self.assertTrue(set_process_priority("above_normal"))
            self.assertEqual(k32.GetPriorityClass(k32.GetCurrentProcess()), 0x00008000)
        finally:
            k32.SetPriorityClass(k32.GetCurrentProcess(), before)


if __name__ == "__main__":
    unittest.main()
