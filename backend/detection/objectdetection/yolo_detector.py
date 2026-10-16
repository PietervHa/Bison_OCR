"""
YOLO Object Detection Module

Provides functions to run YOLOv8 inference for object detection and Region of 
Interest (ROI) extraction.
"""

import os
import threading
import time
import cv2
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)

# Reduce Ultralytics console noise (startup banner/verbose logs).
os.environ.setdefault("YOLO_VERBOSE", "False")

# YOLO model is loaded lazily (only when needed)
_model = None
_model_lock = threading.Lock()
_model_loaded = False


def _get_yolo_model():
    """Lazy load YOLO model on first use."""
    global _model, _model_loaded

    if _model_loaded and _model is None:
        # Already tried to load and failed
        return None

    if _model is not None:
        # Already loaded
        return _model

    try:
        from ultralytics import YOLO
        od_cfg = cfg["object_detection"]
        _model = YOLO(od_cfg["model_path"])  # nano = fast, CPU friendly
        _model_loaded = True
        log.info("YOLO model loaded: model_path=%s", od_cfg["model_path"])
        return _model
    except Exception as exc:
        log.error("Failed to load YOLO model: %s", exc)
        _model_loaded = True
        return None


def extract_rois(frame):
    """Extract regions of interest (ROIs) from frame using YOLO object detection.

    Returns a list of dicts, each containing:
    - "crop": numpy array (BGR) of the detected region
    - "label": class label string from model.names
    - "confidence": float rounded to 3 decimal places
    - "bbox": [x1, y1, x2, y2] in pixels of the original frame
    """
    start = time.perf_counter()

    try:
        # Lazy load YOLO model on first use
        with _model_lock:
            model = _get_yolo_model()
            if model is None:
                raise RuntimeError("YOLO model failed to load")

        # Optional: resize for speed (recommended)
        od_cfg = cfg["object_detection"]
        inference_size = od_cfg["inference_size"]
        frame_resized = cv2.resize(frame, (inference_size, inference_size))
        frame_h, frame_w = frame.shape[:2]

        # Run YOLO inference (guard shared model access across threads)
        with _model_lock:
            results = model(frame_resized, verbose=False)

        rois = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]

                # Get bounding box in resized coordinates, then scale back to original frame
                x1_resized, y1_resized, x2_resized, y2_resized = map(int, box.xyxy[0].tolist())

                # Scale back to original frame coordinates
                scale_w = frame_w / inference_size
                scale_h = frame_h / inference_size
                x1 = int(x1_resized * scale_w)
                y1 = int(y1_resized * scale_h)
                x2 = int(x2_resized * scale_w)
                y2 = int(y2_resized * scale_h)

                # Clamp to frame bounds
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y2 = max(0, min(y2, frame_h - 1))

                # Ensure valid crop dimensions
                if x1 >= x2 or y1 >= y2:
                    continue

                # Extract crop from original frame
                crop = frame[y1:y2, x1:x2]

                rois.append({
                    "crop": crop,
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.debug(
            "ROI extraction: found %s regions in %sms",
            len(rois),
            duration_ms,
        )

        return rois
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.error("ROI extraction failed: %s", exc)
        return []


def run_object_detection(frame):
    """DEPRECATED: Use extract_rois() instead.

    This function is kept for backward compatibility and calls extract_rois()
    internally, returning results in the old format.
    """
    start = time.perf_counter()

    rois = extract_rois(frame)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    # Convert ROIs to old detection format (label + confidence only)
    detections = [
        {
            "label": roi["label"],
            "confidence": roi["confidence"]
        }
        for roi in rois
    ]

    return {
        "detections": detections,
        "processing_time_ms": duration_ms,
        "mode": "object_detection"
    }

