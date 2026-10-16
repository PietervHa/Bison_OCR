import os
import secrets
import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from backend.core.config_loader import cfg
import time


# Seconds a wrong maintenance password is held before the reply goes out,
# to slow down guessing.
_WRONG_PASSWORD_DELAY_S = 1.0


def _maintenance_password() -> str:
    """The maintenance password from MAINTENANCE_PASSWORD (.env or the
    environment). Empty means not configured: maintenance stays locked."""
    return os.environ.get("MAINTENANCE_PASSWORD", "")


def _wrong_password_delay() -> None:
    time.sleep(_WRONG_PASSWORD_DELAY_S)


# --- Request bodies -------------------------------------------------------
# Pydantic models replace Flask's `request.json or {}` pattern. Defaults
# mirror the old `data.get(key, default)` calls exactly.

class MaintenanceModeBody(BaseModel):
    maintenance_mode: bool = False
    password: str = ""


def create_app(camera, app_state) -> FastAPI:
    app = FastAPI()

    # flask-cors' CORS(app) defaults to allowing all origins/methods/headers
    # without credentials; this mirrors that.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    JPEG_QUALITY = int(cfg.get("hmi", {}).get("stream_quality", 75))

    def _is_maintenance_access(request: Request) -> bool:
        if not app_state.get_maintenance_mode():
            return False

        cookie_token = request.cookies.get("maintenance_session", "")
        return bool(cookie_token) and cookie_token == app_state.get_maintenance_session_token()

    def generate_frames():
        if not cfg["hmi"]["enable_video_feed"]:
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Video feed disabled", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # quality applies to the disabled-feed placeholder too
            _, buffer = cv2.imencode(".jpg", blank,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(1)

        FRAME_INTERVAL_MS = 33.33

        while True:
            frame_start = time.time()

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # encode with configured quality instead of OpenCV default (95)
            _, buffer = cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            elapsed_ms = (time.time() - frame_start) * 1000
            remaining_ms = FRAME_INTERVAL_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

    # Applied to the index page and the /status auth check so that no
    # browser, proxy, or embedded kiosk webview caches a response that
    # reflects maintenance access. Without this, some webviews can satisfy
    # a back-button navigation straight from disk/HTTP cache instead of
    # hitting the network, bypassing the pageshow/bfcache reload in hmi.js.
    _NO_STORE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    @app.get("/")
    def index():
        template_path = Path(__file__).resolve().parent / "templates" / "hmi.html"
        return FileResponse(template_path, headers=_NO_STORE_HEADERS)

    @app.get("/status")
    def get_status(request: Request, response: Response):
        response.headers.update(_NO_STORE_HEADERS)
        is_maintenance = _is_maintenance_access(request)
        return {
            "maintenance_mode": is_maintenance,
            "machine_id": cfg["machine_id"],
            "version": "1.0.0",
        }

    @app.get("/result")
    def get_result():
        return app_state.get_snapshot()

    @app.post("/maintenance_mode")
    def set_maintenance_mode(body: MaintenanceModeBody, request: Request, response: Response):
        # Entering maintenance mode requires the single password from
        # MAINTENANCE_PASSWORD. When it isn't configured, maintenance mode
        # stays locked - it is never open by default.
        if body.maintenance_mode:
            expected = _maintenance_password()
            if not expected:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "not_configured",
                        "message": "Maintenance mode is locked: MAINTENANCE_PASSWORD is not set.",
                    },
                )

            if not secrets.compare_digest(body.password.encode("utf-8"), expected.encode("utf-8")):
                _wrong_password_delay()
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid_password", "message": "Incorrect password."},
                )

            session_token = secrets.token_urlsafe(24)
            app_state.set_maintenance_mode(True)
            app_state.set_maintenance_session_token(session_token)
            response.set_cookie(
                "maintenance_session",
                session_token,
                httponly=True,
                samesite="lax",
                max_age=60 * 30,
                path="/",
            )
            return {"maintenance_mode": True}

        app_state.set_maintenance_mode(False)
        app_state.set_maintenance_session_token("")
        response.delete_cookie("maintenance_session", path="/")
        return {"maintenance_mode": False}

    @app.post("/camera_rotation")
    def rotate_camera(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.rotate_camera()
        return {"camera_rotation": app_state.get_camera_rotation()}

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/camera_status")
    def camera_status(response: Response):
        # Polled by the HMI to drive the "reconnecting..." overlay and the
        # "missing camera" popup. No auth required - it's read-only status.
        response.headers.update(_NO_STORE_HEADERS)
        return {"connected": camera.is_connected()}

    @app.post("/reset_counters")
    def reset_counters(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.reset_counters()
        return {"status": "counters reset"}

    return app
