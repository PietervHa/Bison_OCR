"""
Main Entry Point

This module orchestrates the entire vision application, including camera
management, vision processing, the TCP trigger server for PLC integration,
and the web interface.
"""

import threading
import keyboard
import uvicorn
from pathlib import Path

from backend.core.camera import Camera
from backend.core.image_source import ImageFolderSource
from backend.core.vision import run_vision, bind_app_state
from backend.core.tcp_trigger_server import TCPTriggerServer
from frontend.web import create_app
from backend.core.state import AppState
import time
from backend.core.config_loader import cfg
from backend.utils.logger import setup_logging, get_logger, UVICORN_LOG_CONFIG
from backend.output.result_writer import save_result

log = get_logger(__name__)
_vision_busy = threading.Event()

def _process_vision_result(result, trigger_time, app_state):
    """
    Callback to handle vision results from background thread.

    Calculates cycle time, updates the application state, increments counters,
    and saves the result to disk.
    """
    try:
        # Calculate total cycle time from trigger to result
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 1)

        status = result["status"]
        confidence = result.get("confidence", 0)

        result_dict = {
            **result,
            "cycle_time_ms": cycle_time_ms  # Total time from trigger to result
        }
        app_state.update_result(result_dict)
        app_state.increment_counter(status)

        # Deferred I/O: write outside main thread if possible
        try:
            save_result(result_dict)
        except Exception as exc:
            log.error("Failed to write result file: %s", exc)

        detection_count = len(result.get("detections", []))
        failure_reason = result.get("failure_reason")

        if failure_reason:
            log.warning(
                "VISION RESULT: status=%s reason=%s cycle_time_ms=%s confidence=%s detections=%s",
                status,
                failure_reason,
                cycle_time_ms,
                confidence,
                detection_count,
            )
        else:
            log.info(
                "VISION RESULT: status=%s cycle_time_ms=%s confidence=%s detections=%s",
                status,
                cycle_time_ms,
                confidence,
                detection_count,
            )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)

def start_vision_cycle(camera, app_state) -> bool:
    """Grab a frame and start one vision cycle in the background.
    Returns False when a cycle is already running."""
    global _vision_busy
    if _vision_busy.is_set():
        return False
    frame = camera.get_frame()
    if frame is None:
        log.warning("No frame available; reporting NOK")

    # Track trigger time for cycle time measurement
    trigger_time = time.perf_counter()

    def _callback(result):
        try:
            _process_vision_result(result, trigger_time, app_state)
        finally:
            _vision_busy.clear()

    _vision_busy.set()
    try:
        run_vision(frame, callback=_callback)  # background thread, calls _callback
    except Exception:
        _vision_busy.clear()
        raise
    log.debug("Vision processing started (non-blocking)")
    return True


def vision_trigger_loop(camera, app_state):
    """
    Manual trigger loop that listens for keyboard input to start a vision cycle.

    Pressing 'q' triggers the camera to capture a frame and starts vision
    processing in a background thread.
    """
    log.info("Press Q to trigger vision. Ctrl+C to exit.")

    while True:
        try:
            keyboard.wait("q")
            start_vision_cycle(camera, app_state)
        except Exception as exc:
            log.error("Vision trigger loop error: %s", exc)


def make_frame_source(app_state):
    """The camera, or still images from a folder (camera.source: "images")."""
    cam_cfg = cfg.get("camera", {}) or {}
    if str(cam_cfg.get("source", "live")).lower() == "images":
        folder = Path(cam_cfg.get("images_dir") or "data/test_images")
        if not folder.is_absolute():
            folder = Path(__file__).resolve().parents[1] / folder
        log.info("Frame source: test images from %s (arrow keys on the HMI)", folder)
        return ImageFolderSource(folder, app_state=app_state,
                                 max_width=cam_cfg.get("images_max_width", 744))
    return Camera(0, app_state=app_state)


def main():
    """
    Initializes and starts all core components of the application.

    Sets up logging, application state, camera, web server, and TCP trigger server.
    """
    setup_logging()
    app_state = AppState()
    bind_app_state(app_state)

    try:
        camera = make_frame_source(app_state)

        web_cfg = cfg["web"]
        app = create_app(camera, app_state, start_cycle=lambda: start_vision_cycle(camera, app_state))
        web_thread = threading.Thread(
            target=lambda: uvicorn.run(
                app,
                host=web_cfg["host"],
                port=web_cfg["port"],
                log_config=UVICORN_LOG_CONFIG,
            ),
            daemon=True,
        )
        web_thread.start()

        display_host = "127.0.0.1" if web_cfg["host"] in ("0.0.0.0", "127.0.0.1", "localhost") else web_cfg["host"]
        log.info("Web dashboard: http://%s:%s", display_host, web_cfg["port"])

        trigger_server = TCPTriggerServer(
            camera=camera,
            run_vision_fn=run_vision,
            process_result_fn=lambda result, t: _process_vision_result(result, t, app_state),
        )
        trigger_server.start()

        vision_trigger_loop(camera, app_state)
    except Exception as exc:
        log.error("Fatal startup/runtime error: %s", exc)
        raise

if __name__ == "__main__":
    main()