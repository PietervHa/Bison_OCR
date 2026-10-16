import unittest
from unittest.mock import patch

from backend.core.state import AppState
from backend.core import inspection_engine


class _FakeState:
    def __init__(self, threshold=0.8):
        self._threshold = threshold

    def get_threshold(self):
        return self._threshold


class _FakeClassifier:
    def __init__(self, prediction, loaded=True):
        self._prediction = prediction
        self._loaded = loaded

    def is_loaded(self):
        return self._loaded

    def predict(self, _image):
        return self._prediction


class AppStateUnitTests(unittest.TestCase):
    def test_threshold_is_clamped_and_counters_are_updated(self):
        state = AppState()

        state.set_threshold(2.0)
        self.assertEqual(state.get_threshold(), 1.0)

        state.set_threshold(-1.0)
        self.assertEqual(state.get_threshold(), 0.0)

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


class InspectionEngineUnitTests(unittest.TestCase):
    def setUp(self):
        self._original_backend = inspection_engine.cfg["object_detection"].get("backend")
        inspection_engine.cfg["object_detection"]["backend"] = "classifier"

    def tearDown(self):
        inspection_engine.cfg["object_detection"]["backend"] = self._original_backend

    def test_parse_detection_item_supports_multiple_label_and_score_keys(self):
        parsed = inspection_engine.InspectionEngine._parse_detection_item(
            {"class_name": "cap", "score": "0.8333"}
        )
        self.assertEqual(parsed["label"], "cap")
        self.assertEqual(parsed["text"], "cap")
        self.assertEqual(parsed["confidence"], 0.833)

    def test_extract_predictions_handles_nested_outputs(self):
        payload = {
            "outputs": {
                "model": {
                    "predictions": [
                        {"class": "ok", "confidence": 0.91},
                        "invalid",
                    ]
                }
            }
        }

        predictions = inspection_engine.InspectionEngine._extract_predictions(payload)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["class"], "ok")

    def test_evaluate_returns_no_model_when_classifier_not_loaded(self):
        engine = inspection_engine.InspectionEngine(_FakeState())

        result = engine.evaluate(frame=None)

        self.assertEqual(result["status"], "NOK")
        self.assertEqual(result["error"], "no_model")
        self.assertEqual(result["detections"], [])

    def test_evaluate_maps_ok_label_to_non_defective(self):
        engine = inspection_engine.InspectionEngine(_FakeState(threshold=0.5))
        engine._classifier = _FakeClassifier(
            {
                "label": "ok",
                "confidence": 0.95,
                "all_scores": {"ok": 0.95, "defective": 0.05},
            }
        )

        with patch("backend.core.inspection_engine.preprocess_to_pil", return_value="pil-image"):
            result = engine.evaluate(frame="frame")

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["label"], "non-defective")
        self.assertAlmostEqual(result["confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()

