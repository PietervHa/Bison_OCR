"""
Vision Orchestration Module

This module is the entry point for vision tasks: it runs the OCRead engine
(backend/detection/ocread/engine.py) on a frame and turns its result into an
OK/NOK status.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from backend.core.config_loader import cfg
from backend.detection.ocread.engine import MODE, OCRead
from backend.utils.logger import get_logger
from backend.utils.priority import set_thread_priority
import time

logger = get_logger(__name__)

# Global instances managed by this module
ocr_instance = None

def _init_worker() -> None:
    """Runs once on the vision thread: its priority
    (performance.vision_thread_priority, see backend/utils/priority.py)."""
    set_thread_priority((cfg.get("performance") or {}).get("vision_thread_priority", "normal"))


# A single-threaded pool to process vision tasks sequentially without blocking the main loop
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision_worker", initializer=_init_worker)


def bind_app_state(app_state) -> None:
    """
    Initializes the OCRead engine. app_state is accepted for the caller's
    convenience; the engine itself needs no application state.
    """
    global ocr_instance
    ocr_instance = OCRead()


def _normalize_detections(result: dict) -> list:
    """
    Ensures that detections are always a list of dicts.
    """
    detections = result.get("detections")
    if isinstance(detections, list):
        normalized = [d for d in detections if isinstance(d, dict)]
    else:
        normalized = []

    return normalized


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


def _normalize_result(result: dict) -> dict:
    """
    Standardizes the result dictionary and decides its OK/NOK status.
    """
    normalized = dict(result or {})
    normalized.setdefault("mode", MODE)
    normalized["detections"] = _normalize_detections(normalized)
    normalized["confidence"] = _extract_confidence(normalized)

    has_error = bool(normalized.get("error"))

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
        # Read did not pass read_validation (format/confidence/value):
        # an explicit reject with its reason, never a guessed OK.
        normalized["status"] = "NOK"
        normalized.setdefault("failure_reason", normalized["validation"].get("reason") or "format_mismatch")
    elif not normalized["text"].strip() and not normalized["detections"]:
        normalized["status"] = "NOK"
        normalized.setdefault("failure_reason", "no_text_detected")
    else:
        normalized["status"] = "OK"
    return normalized


def _run_with_callback(fn, frame, callback, profile: bool = False) -> None:
    """
    Executes a vision function asynchronously in a background thread and returns the result via a callback.
    """
    def worker():
        start = time.perf_counter()
        result = None
        exception_msg = None
        try:
            # The worker runs the read off the main thread so the caller
            # can continue immediately while the result is prepared.
            result = fn(frame, profile=profile) if profile else fn(frame)
        except Exception as exc:
            exception_msg = str(exc)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            result = {
                "status": "NOK",
                "detections": [],
                "processing_time_ms": duration_ms,
                "mode": MODE,
                "error": exception_msg,
            }

        if exception_msg:
            logger.error("Vision worker exception: %s", exception_msg)
        callback(_normalize_result(result))

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
        normalized = _normalize_result(no_frame_result)
        if callback:
            callback(normalized)
            return None
        return normalized

    if callback:
        _run_with_callback(ocr_instance.read, frame, callback, profile=profile)
        return None
    return _normalize_result(ocr_instance.read(frame, profile=profile))
