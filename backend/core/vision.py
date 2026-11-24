"""
Vision Orchestration Module

This module is the entry point for vision tasks: it runs the OCRead engine
(backend/detection/ocread/engine.py) on a frame and turns its result into an
OK/NOK status.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from backend.core import config_loader, profiles
from backend.core.config_loader import cfg
from backend.detection.ocread.engine import MODE, OCRead
from backend.utils.logger import get_logger
from backend.utils.priority import set_thread_priority
import time

logger = get_logger(__name__)

# Global instances managed by this module
ocr_instance = None
# Name of the product profile ocr_instance reads with (backend/core/profiles.py).
active_profile = None

def _init_worker() -> None:
    """Runs once on the vision thread: its priority
    (performance.vision_thread_priority, see backend/utils/priority.py)."""
    set_thread_priority((cfg.get("performance") or {}).get("vision_thread_priority", "normal"))


# A single-threaded pool to process vision tasks sequentially without blocking the main loop
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision_worker", initializer=_init_worker)


def build_engine(name: str) -> OCRead:
    """An OCRead engine for product profile `name` (the config file with the
    profile's ocr overrides), warmed up when ocr.speed.warmup is set.
    Raises profiles.ProfileError for a profile that does not exist or is
    invalid."""
    engine = OCRead(profiles.config_for(name), profile_name=name)
    if engine.speed.get("warmup") and engine.font_reader is not None:
        logger.info("OCRead [%s] warm-up: %.0f ms", name, engine.warmup())
    return engine


def _state_file() -> Path:
    """Where the profile chosen on the HMI is remembered: one file per
    config file, so test runs (default.yaml) and production
    (production.yaml) each keep their own."""
    stem = Path(config_loader.config_path_loaded or "default").stem
    return profiles.PROJECT_ROOT / "data" / f"active_profile_{stem}.txt"


def _remembered_profile():
    try:
        name = _state_file().read_text(encoding="utf-8").strip()
        return name or None
    except OSError:
        return None


def _remember_profile(name: str) -> None:
    if not (cfg.get("profiles") or {}).get("remember", True):
        return
    try:
        path = _state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save the active profile (%s): %s", name, exc)


def startup_profile() -> str:
    """profiles.default, or the profile last chosen on the HMI when
    profiles.remember is on and that profile still exists."""
    p_cfg = cfg.get("profiles") or {}
    name = str(p_cfg.get("default") or profiles.STANDARD)
    if p_cfg.get("remember", True):
        remembered = _remembered_profile()
        if remembered and remembered in profiles.names():
            name = remembered
    return name


def bind_app_state(app_state) -> None:
    """
    Initializes the OCRead engine with the startup profile (startup_profile).
    A startup profile that cannot be loaded or has no usable font reader
    falls back to "standard" (logged as an error). app_state is accepted for
    the caller's convenience; the engine itself needs no application state.
    """
    global ocr_instance, active_profile
    name = startup_profile()
    engine = None
    try:
        engine = build_engine(name)
    except Exception as exc:
        logger.error("Profile %r could not be loaded: %s", name, exc)
    if (engine is None or engine.font_reader is None) and name != profiles.STANDARD:
        logger.error("Profile %r is not usable - starting with profile %r", name, profiles.STANDARD)
        name, engine = profiles.STANDARD, build_engine(profiles.STANDARD)
    ocr_instance, active_profile = engine, name
    logger.info("Product profile: %s", name)


def profile_status() -> dict:
    """The active profile and every profile the HMI can choose from."""
    return {"active": active_profile, "profiles": [profiles.describe(n) for n in profiles.names()]}


def _switch(name: str) -> dict:
    global ocr_instance, active_profile
    engine = build_engine(name)
    if engine.font_reader is None:
        raise profiles.ProfileError(f"profile {name!r} cannot read: {engine.unavailable_reason}")
    previous = active_profile
    ocr_instance, active_profile = engine, name
    _remember_profile(name)
    logger.warning("Product profile switched: %s -> %s", previous, name)
    return profile_status()


def switch_profile(name: str, timeout_s: float = 60.0) -> dict:
    """Switch to product profile `name` without a restart. The new engine is
    built (and warmed up) on the vision thread itself, queued like a read:
    a read that is running finishes with the old profile, every read
    started after it uses the new one, and no read ever runs during the
    swap. A profile that cannot be loaded or cannot read (model missing,
    no supported format) raises profiles.ProfileError and the active
    profile stays as it was. Returns profile_status()."""
    return _pool.submit(_switch, name).result(timeout=timeout_s)


def _read(frame, **kwargs):
    """The active engine's read, looked up when the read starts (not when it
    was queued): a read queued behind a profile switch uses the new profile."""
    return ocr_instance.read(frame, **kwargs)


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


def capture_time_of(source):
    """The capture time of the frame source's last frame (its
    frame_capture_time(): the system clock for the camera, the file name for
    test images), None when the source has none."""
    fn = getattr(source, "frame_capture_time", None)
    try:
        return fn() if callable(fn) else None
    except Exception as exc:  # pragma: no cover - never let this stop a cycle
        logger.warning("Capture time unavailable: %s", exc)
        return None


def _run_with_callback(fn, frame, callback, profile: bool = False, capture_time=None) -> None:
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
            kwargs = {"profile": True} if profile else {}
            if capture_time is not None:
                kwargs["capture_time"] = capture_time
            result = fn(frame, **kwargs)
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


def run_vision(frame, callback=None, profile: bool = False, capture_time=None):
    """
    The main entry point for running a vision task on a frame.
    
    Supports both synchronous and asynchronous (via callback) execution.
    capture_time: when the frame was captured (capture_time_of(source)), for
    OCRead's time plausibility check; None = unknown.
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
        _run_with_callback(_read, frame, callback, profile=profile, capture_time=capture_time)
        return None
    kwargs = {"capture_time": capture_time} if capture_time is not None else {}
    return _normalize_result(ocr_instance.read(frame, profile=profile, **kwargs))
