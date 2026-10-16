from __future__ import annotations
import numpy as np
from PIL import Image
from backend.core.inspection_engine import InspectionEngine
from backend.core.state import AppState
from backend.detection.objectdetection.classifier import ImageClassifier
from backend.detection.objectdetection.preprocessing import preprocess, preprocess_to_pil


TOTAL_TESTS = 5


def _make_fake_frame() -> np.ndarray:
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


def _run_test(test_name: str, test_fn) -> bool:
    try:
        test_fn()
        return True
    except Exception as exc:
        print(f"FAIL: {test_name} — {exc}")
        return False


def test_1_preprocessing() -> None:
    frame = _make_fake_frame()
    out = preprocess(frame, size=224)
    assert out.shape == (224, 224, 3), f"unexpected shape: {out.shape}"
    assert out.dtype == np.uint8, f"unexpected dtype: {out.dtype}"
    print("PASS: preprocessing output shape and dtype correct")


def test_2_preprocess_to_pil() -> None:
    frame = _make_fake_frame()
    pil_image = preprocess_to_pil(frame, size=224)
    assert isinstance(pil_image, Image.Image), f"unexpected type: {type(pil_image)!r}"
    assert pil_image.size == (224, 224), f"unexpected size: {pil_image.size}"
    assert pil_image.mode == "RGB", f"unexpected mode: {pil_image.mode}"
    print("PASS: preprocess_to_pil returns correct PIL image")


def test_3_classifier_no_model() -> None:
    classifier = ImageClassifier(model_path="models/nonexistent/")
    assert classifier.is_loaded() is False, "classifier unexpectedly loaded"
    print("PASS: classifier handles missing model path gracefully")


def test_4_inspection_no_model_fallback() -> None:
    frame = _make_fake_frame()

    class MockAppState:
        @staticmethod
        def get_threshold() -> float:
            return 0.8

    engine = InspectionEngine(MockAppState())
    result = engine.evaluate(frame)
    assert result["status"] == "NOK", f"unexpected status: {result}"
    assert result["error"] == "no_model", f"unexpected error: {result}"
    print("PASS: InspectionEngine returns NOK safely when no model loaded")


def test_5_app_state_classifier_status() -> None:
    app_state = AppState()

    status = app_state.get_classifier_status()
    assert status == {"loaded": False, "model_path": ""}, f"unexpected initial status: {status}"

    app_state.set_classifier_loaded("models/test/")
    status = app_state.get_classifier_status()
    assert status == {"loaded": True, "model_path": "models/test/"}, f"unexpected updated status: {status}"

    print("PASS: AppState classifier status tracking works")


def main() -> None:
    tests = [
        ("TEST 1 - Preprocessing", test_1_preprocessing),
        ("TEST 2 - Preprocessing to PIL", test_2_preprocess_to_pil),
        ("TEST 3 - Classifier graceful no-model", test_3_classifier_no_model),
        ("TEST 4 - InspectionEngine no-model fallback", test_4_inspection_no_model_fallback),
        ("TEST 5 - AppState classifier status", test_5_app_state_classifier_status),
    ]

    passed = 0
    for test_name, test_fn in tests:
        if _run_test(test_name, test_fn):
            passed += 1

    print(f"{passed}/{TOTAL_TESTS} tests passed")


if __name__ == "__main__":
    main()

