"""
Vision Orchestration Module

This module serves as the primary entry point for vision tasks, running the OCR
engine on a frame and managing the communication with the application state.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from backend.core.config_loader import cfg
from backend.detection.ocr import OCR
from backend.utils.logger import get_logger
import time

logger = get_logger(__name__)

# Backward-compatible alias; primary source is cfg["vision_mode"].
VISION_MODE = cfg["vision_mode"]

# Global instances managed by this module
ocr_instance = None
_app_state: Optional[Any] = None
_cached_vision_mode: str = cfg.get("vision_mode", "ocread")

# A single-threaded pool to process vision tasks sequentially without blocking the main loop
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision_worker")


def bind_app_state(app_state) -> None:
    """
    Connects the vision module to the application state and initializes the OCR engine.
    """
    global ocr_instance, _app_state, _cached_vision_mode
    _app_state = app_state
    _cached_vision_mode = cfg.get("vision_mode", "ocread")
    ocr_instance = OCR(app_state=app_state)
    if hasattr(ocr_instance, "warmup"):
        ocr_instance.warmup()


def _normalize_detections(result: dict) -> list:
    """
    Ensures that detection results have a consistent format across different backends.
    """
    # Every backend should expose detections in a consistent list shape so the
    # UI and PLC logic do not need backend-specific branches.
    detections = result.get("detections")
    if isinstance(detections, list):
        normalized = [d for d in detections if isinstance(d, dict)]
    else:
        normalized = []

    return normalized


def _get_active_threshold() -> float:
    """
    Retrieves the current confidence threshold from app state or configuration.
    """
    app_state = _app_state
    if app_state is not None and hasattr(app_state, "get_threshold"):
        try:
            return float(app_state.get_threshold())
        except Exception:
            pass
    return float(cfg.get("confidence_threshold", 0.0))


def _extract_confidence(normalized: dict) -> float:
    """
    Determines the overall confidence for a result, usually the highest confidence among detections.
    """
    if "confidence" in normalized:
        try:
            return float(normalized.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    detections = normalized.get("detections") or []
    best = 0.0
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        try:
            conf = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf > best:
            best = conf
    return best


def _normalize_result(result: dict, mode: str) -> dict:
    """
    Standardizes the result dictionary, calculating status based on confidence and thresholds.
    """
    normalized = dict(result or {})
    normalized.setdefault("mode", mode)
    normalized["detections"] = _normalize_detections(normalized)
    normalized["confidence"] = _extract_confidence(normalized)

    has_error = bool(normalized.get("error"))

    if mode == "ocread":
        # OCRead's own status just reflects whether the product should be
        # accepted or discarded before it even reaches the PLC comparison:
        #   - no text read at all -> NOK, so the product gets rejected
        #     immediately (bad print, empty/blank part, camera out of focus,
        #     wrong angle, etc.)
        #   - text was read (any text) -> OK; the PLC still does its own
        #     comparison against the expected value.
        # A technical failure (has_error) is NOT the same thing as "no text
        # found" - it means the read cycle itself couldn't run - so it's
        # kept NOK too but tagged with failure_reason for separate logging,
        # the same convention used for camera_unavailable/vision_timeout/etc.
        normalized.setdefault("text", "")
        if has_error:
            normalized["status"] = "NOK"
            normalized.setdefault("failure_reason", "ocr_engine_error")
        elif isinstance(normalized.get("validation"), dict) and not normalized["validation"].get("accepted", True):
            # Read did not pass read_validation (format/confidence/consensus):
            # an explicit reject with its reason, never a guessed OK.
            normalized["status"] = "NOK"
            normalized.setdefault("failure_reason", normalized["validation"].get("reason") or "format_mismatch")
        elif not normalized["text"].strip() and not normalized["detections"]:
            normalized["status"] = "NOK"
            normalized.setdefault("failure_reason", "no_text_detected")
        else:
            normalized["status"] = "OK"
        return normalized

    threshold = _get_active_threshold()

    if has_error:
        normalized["status"] = "NOK"
        return normalized

    # Threshold is leading: confidence below threshold is always NOK.
    if normalized["confidence"] < threshold:
        normalized["status"] = "NOK"
        return normalized

    if "status" not in normalized:
        normalized["status"] = "OK" if normalized["detections"] else "NOK"

    return normalized


def _run_with_callback(fn, frame, callback, mode: str, profile: bool = False) -> None:
    """
    Executes a vision function asynchronously in a background thread and returns the result via a callback.
    """
    def worker():
        start = time.perf_counter()
        result = None
        exception_msg = None
        try:
            # The worker runs the selected backend off the main thread so the
            # caller can continue immediately while the result is prepared.
            result = fn(frame, profile=profile) if profile else fn(frame)
        except Exception as exc:
            exception_msg = str(exc)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            result = {
                "status": "NOK",
                "detections": [],
                "processing_time_ms": duration_ms,
                "mode": mode,
                "error": exception_msg,
            }

        if exception_msg:
            logger.error("Vision worker exception: %s", exception_msg)
        callback(_normalize_result(result, mode))

    _pool.submit(worker)


def run_vision(frame, callback=None, profile: bool = False):
    """
    The main entry point for running a vision task on a frame.
    
    Supports both synchronous and asynchronous (via callback) execution.
    """
    if frame is None:
        no_frame_result = {
            "status": "NOK",
            "detections": [],
            "processing_time_ms": 0,
            "error": "no_frame",
        }
        mode_getter = getattr(_app_state, "get_vision_mode", None)
        mode = mode_getter() if callable(mode_getter) else _cached_vision_mode
        normalized = _normalize_result(no_frame_result, mode)
        if callback:
            callback(normalized)
            return None
        return normalized

    # The active mode is resolved from app state first so runtime changes win
    # over the cached startup configuration.
    mode_getter = getattr(_app_state, "get_vision_mode", None)
    mode = mode_getter() if callable(mode_getter) else _cached_vision_mode

    if mode == "ocr":
        if callback:
            _run_with_callback(ocr_instance.run, frame, callback, mode="ocr", profile=profile)
            return None
        return _normalize_result(ocr_instance.run(frame, profile=profile), mode="ocr")

    if mode == "ocread":
        if callback:
            _run_with_callback(ocr_instance.read, frame, callback, mode="ocread", profile=profile)
            return None
        return _normalize_result(ocr_instance.read(frame, profile=profile), mode="ocread")

    raise ValueError(f"Unknown VISION_MODE: {mode}")