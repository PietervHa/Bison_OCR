"""
Inspection Engine

This module provides the InspectionEngine class which abstracts various object 
detection and classification backends (Local Classifier, Roboflow, YOLO, Template Matching).
"""

from __future__ import annotations
import os
import tempfile
import time
import cv2
from pathlib import Path
from backend.core.config_loader import cfg
from backend.detection.objectdetection.preprocessing import preprocess_to_pil
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class InspectionEngine:
    """
    Orchestrates the inspection process using a selected detection backend.
    """
    def __init__(self, app_state):
        self.app_state = app_state
        self._classifier = None
        self._roboflow_client = None
        self._template_detector = None

        od_cfg = cfg.get("object_detection", {})
        # The backend can switch between local and remote detectors without
        # changing the rest of the inspection pipeline.
        backend = str(od_cfg.get("backend", "classifier")).strip().lower()
        if backend not in {"classifier", "roboflow", "yolo", "template"}:
            logger.warning("Unknown detector backend '%s'; falling back to 'classifier'", backend)
            backend = "classifier"

        self._detector_backend = backend
        logger.info("Object detection backend selected: %s", self._detector_backend)
        roboflow_cfg = od_cfg.get("roboflow", {}) if isinstance(od_cfg.get("roboflow"), dict) else {}
        self._roboflow_model = str(roboflow_cfg.get("model", "default")).strip() or "default"
        self._roboflow_workspace = str(roboflow_cfg.get("workspace", "")).strip()
        self._roboflow_workflow = str(roboflow_cfg.get("workflow", "")).strip()
        self._roboflow_api_url = str(roboflow_cfg.get("api_url", "https://serverless.roboflow.com")).strip()

        if self._detector_backend == "roboflow":
            self._init_roboflow_detector()
        if self._detector_backend == "template":
            self._init_template_detector()

    def _init_template_detector(self):
        """
        Initializes the template matching detector with references from configuration.
        """
        od_cfg = cfg.get("object_detection", {})
        template_cfg = od_cfg.get("template", {}) if isinstance(od_cfg.get("template"), dict) else {}
        reference_paths = list(template_cfg.get("references", []))
        match_threshold = float(template_cfg.get("match_threshold", 0.6))
        display_label = str(template_cfg.get("display_label", "")).strip()

        resolved_paths = []
        for reference_path in reference_paths:
            path_obj = Path(reference_path)
            if not path_obj.is_absolute():
                path_obj = Path(__file__).resolve().parents[2] / path_obj
            resolved_paths.append(str(path_obj.resolve()))

        try:
            from backend.detection.objectdetection.template_detector import TemplateDetector

            self._template_detector = TemplateDetector(
                resolved_paths,
                match_threshold=match_threshold,
                display_label=display_label,
            )
            logger.info(
                "Template detector initialized successfully: references=%s match_threshold=%.3f label=%s",
                len(resolved_paths),
                match_threshold,
                display_label,
            )
        except Exception as exc:
            logger.error("Failed to initialize template detector: %s", exc)
            self._template_detector = None

    def _init_roboflow_detector(self) -> None:
        """
        Initializes the Roboflow Inference client.
        """
        od_cfg = cfg.get("object_detection", {})
        roboflow_cfg = od_cfg.get("roboflow", {}) if isinstance(od_cfg.get("roboflow"), dict) else {}
        api_key = str(os.environ.get("ROBOFLOW_API_KEY", "")).strip()

        if not api_key or not self._roboflow_workspace or not self._roboflow_workflow:
            logger.warning("Roboflow configuration is incomplete; detector will stay disabled")
            self._roboflow_client = None
            return

        try:
            from inference_sdk import InferenceHTTPClient

            init_fn = getattr(InferenceHTTPClient, "init", None)
            if callable(init_fn):
                self._roboflow_client = init_fn(
                    api_url=self._roboflow_api_url,
                    api_key=api_key,
                )
            else:
                self._roboflow_client = InferenceHTTPClient(
                    api_url=self._roboflow_api_url,
                    api_key=api_key,
                )

            logger.info("Roboflow detector initialized successfully: model=%s workspace=%s workflow=%s", self._roboflow_model, self._roboflow_workspace, self._roboflow_workflow)
        except Exception as exc:
            logger.error("Failed to initialize Roboflow detector: %s", exc)
            self._roboflow_client = None

    def load_classifier(self, model_path: str) -> bool:
        """
        Loads a local ImageClassifier model.
        """
        if self._detector_backend == "roboflow":
            return self._roboflow_client is not None
        if self._detector_backend == "yolo":
            # YOLO backend does not consume classifier model paths.
            return False

        try:
            from backend.detection.objectdetection.classifier import ImageClassifier

            classifier = ImageClassifier(model_path=model_path)
            if not classifier.is_loaded():
                logger.warning("Classifier could not be loaded: model_path=%s", model_path)
                self._classifier = None
                return False

            self._classifier = classifier
            classifier.warmup()
            return True
        except Exception as exc:
            logger.error("Failed to load classifier: model_path=%s error=%s", model_path, exc)
            self._classifier = None
            return False

    @staticmethod
    def _parse_detection_item(item: dict) -> dict:
        """
        Extracts label and confidence from various detection item formats.
        """
        label = str(
            item.get("class")
            or item.get("class_name")
            or item.get("label")
            or item.get("name")
            or "unknown"
        )

        raw_conf = item.get("confidence", item.get("score", 0.0))
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "label": label,
            "text": label,
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def _extract_predictions(payload: dict) -> list:
        """
        Parses Roboflow response payloads to find the predictions list.
        """
        # Roboflow-style responses are not always shaped the same, so we
        # inspect the common variants before giving up.
        direct = payload.get("predictions")
        if isinstance(direct, list):
            return [p for p in direct if isinstance(p, dict)]
        if isinstance(direct, dict):
            nested = direct.get("predictions")
            if isinstance(nested, list):
                return [p for p in nested if isinstance(p, dict)]

        outputs = payload.get("outputs")
        if isinstance(outputs, dict):
            for value in outputs.values():
                if isinstance(value, dict) and isinstance(value.get("predictions"), list):
                    return [p for p in value["predictions"] if isinstance(p, dict)]
                if isinstance(value, list):
                    return [p for p in value if isinstance(p, dict)]
        if isinstance(outputs, list):
            for value in outputs:
                if isinstance(value, dict) and isinstance(value.get("predictions"), list):
                    return [p for p in value["predictions"] if isinstance(p, dict)]

        return []

    def _run_roboflow_workflow(self, frame):
        """
        Sends a frame to Roboflow for workflow execution.
        """
        if self._roboflow_client is None:
            raise RuntimeError("roboflow_detector_unavailable")

        # The SDK expects a file path, so we write the frame to a temporary
        # image and clean it up immediately after the request finishes.
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        try:
            if frame is None:
                raise ValueError("empty_frame")
            if not cv2.imwrite(temp_path, frame):
                raise RuntimeError("failed_to_encode_frame")

            return self._roboflow_client.run_workflow(
                workspace_name=self._roboflow_workspace,
                workflow_id=self._roboflow_workflow,
                images={"image": temp_path},
                use_cache=False,
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _evaluate_with_roboflow(self, frame, start_time: float) -> dict:
        """
        Executes Roboflow workflow and parses the results.
        """
        raw_result = self._run_roboflow_workflow(frame)

        payload: dict
        if isinstance(raw_result, list):
            payload = raw_result[0] if raw_result and isinstance(raw_result[0], dict) else {}
        elif isinstance(raw_result, dict):
            payload = raw_result
        else:
            raise TypeError(f"Unsupported Roboflow response type: {type(raw_result).__name__}")

        predictions = self._extract_predictions(payload)
        detections = [self._parse_detection_item(item) for item in predictions]
        best_confidence = max((d.get("confidence", 0.0) for d in detections), default=0.0)

        try:
            threshold = float(self.app_state.get_threshold())
        except Exception:
            threshold = 0.0

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Deferred logging: only log at debug level if enabled
        if logger.isEnabledFor(10):  # logging.DEBUG
            logger.debug(
                "Roboflow parsed payload: model=%s keys=%s predictions=%s best_confidence=%.3f threshold=%.3f",
                self._roboflow_model,
                sorted(payload.keys()) if isinstance(payload, dict) else [],
                len(predictions),
                best_confidence,
                threshold,
            )

        return {
            "status": "OK" if detections and best_confidence >= threshold else "NOK",
            "detections": detections,
            "confidence": round(best_confidence, 3),
            "processing_time_ms": processing_time_ms,
            "mode": "object_detection",
        }

    def evaluate(self, frame) -> dict:
        """
        Main evaluation entry point. Routes to the appropriate backend.
        """
        start_time = time.perf_counter()

        if self._detector_backend == "roboflow":
            try:
                return self._evaluate_with_roboflow(frame, start_time)
            except Exception as exc:
                processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error("Roboflow evaluation failed: %s", exc)
                return {
                    "status": "NOK",
                    "error": str(exc),
                    "detections": [],
                    "processing_time_ms": processing_time_ms,
                    "mode": "object_detection",
                }

        if self._detector_backend == "yolo":
            try:
                from backend.detection.objectdetection import run_object_detection

                # YOLO shares the same output contract as the other detectors,
                # so callers can treat all backends uniformly.
                result = run_object_detection(frame)
                result.setdefault("mode", "object_detection")
                return result
            except Exception as exc:
                processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error("YOLO evaluation failed: %s", exc)
                return {
                    "status": "NOK",
                    "error": str(exc),
                    "detections": [],
                    "processing_time_ms": processing_time_ms,
                    "mode": "object_detection",
                }

        if self._detector_backend == "template":
            try:
                result = self._template_detector.detect(frame)
                result.setdefault("mode", "object_detection")
                return result
            except Exception as exc:
                processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error("Template evaluation failed: %s", exc)
                return {
                    "status": "NOK",
                    "error": str(exc),
                    "detections": [],
                    "processing_time_ms": processing_time_ms,
                    "mode": "object_detection",
                }

        if self._classifier is None or not self._classifier.is_loaded():
            return {
                "status": "NOK",
                "error": "no_model",
                "detections": [],
                "processing_time_ms": 0,
            }

        try:
            pil_image = preprocess_to_pil(frame, size=224)
            prediction = self._classifier.predict(pil_image)
            threshold = self.app_state.get_threshold()

            # Map classifier label "ok" to "non-defective" for display
            label = prediction["label"]
            if label.lower() == "ok":
                label = "non-defective"

            # Determine status based on threshold and confidence (ok/non-defective = OK status)
            status = "OK" if prediction["confidence"] >= threshold and prediction["label"].lower() == "ok" else "NOK"
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

            return {
                "status": status,
                "label": label,
                "confidence": prediction["confidence"],
                "all_scores": prediction["all_scores"],
                "processing_time_ms": processing_time_ms,
                "mode": "object_detection",
            }
        except Exception as exc:
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 3)
            logger.error("Inspection evaluation failed: %s", exc)
            return {
                "status": "NOK",
                "error": str(exc),
                "detections": [],
                "processing_time_ms": processing_time_ms,
            }

    def has_model(self) -> bool:
        """
        Checks if the current detector backend is properly initialized/loaded.
        """
        if self._detector_backend == "roboflow":
            return self._roboflow_client is not None
        if self._detector_backend == "yolo":
            return True
        if self._detector_backend == "template":
            return self._template_detector is not None and self._template_detector.is_loaded()
        return self._classifier is not None and self._classifier.is_loaded()

