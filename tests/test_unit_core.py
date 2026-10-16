import unittest

from backend.core.state import AppState


class AppStateUnitTests(unittest.TestCase):
    def test_counters_are_updated(self):
        state = AppState()

        state.increment_counter("OK")
        state.increment_counter("NOK")
        state.increment_counter("UNKNOWN")

        self.assertEqual(state.counters["total"], 3)
        self.assertEqual(state.counters["ok"], 1)
        self.assertEqual(state.counters["nok"], 1)

    def test_snapshot_returns_copied_top_level_dicts(self):
        state = AppState()
        state.update_result({"status": "OK", "detections": []})

        snapshot = state.get_snapshot()
        snapshot["result"]["status"] = "NOK"
        snapshot["counters"]["ok"] = 999

        self.assertEqual(state.latest_result["status"], "OK")
        self.assertNotEqual(state.counters["ok"], 999)


if __name__ == "__main__":
    unittest.main()
