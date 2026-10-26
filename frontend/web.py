import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from backend.core.config_loader import cfg
import time


def create_app(camera, app_state, start_cycle=None) -> FastAPI:
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
    # reflects the current state. Without this, some webviews can satisfy
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
        return {
            "machine_id": cfg["machine_id"],
            "version": "1.0.0",
        }

    @app.get("/result")
    def get_result():
        return app_state.get_snapshot()


    @app.post("/camera_rotation")
    def rotate_camera(request: Request):
        app_state.rotate_camera()
        return {"camera_rotation": app_state.get_camera_rotation()}

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # --- Test images (camera.source: "images") ------------------------------
    # The HMI's arrow keys step through the images; each step runs a read.
    @app.get("/test_images")
    def test_images_status():
        return camera.status() if hasattr(camera, "step") else {"source": "live"}

    def _step_test_image(delta):
        if not hasattr(camera, "step"):
            return Response(status_code=409, content="Not in test-image mode (camera.source is live)")
        status = camera.step(delta)
        status["read_started"] = bool(start_cycle()) if start_cycle and status["count"] else False
        return status

    @app.post("/test_images/next")
    def test_images_next():
        return _step_test_image(+1)

    @app.post("/test_images/prev")
    def test_images_prev():
        return _step_test_image(-1)

    @app.get("/camera_status")
    def camera_status(response: Response):
        # Polled by the HMI to drive the "reconnecting..." overlay and the
        # "missing camera" popup. No auth required - it's read-only status.
        response.headers.update(_NO_STORE_HEADERS)
        return {"connected": camera.is_connected()}

    @app.post("/reset_counters")
    def reset_counters(request: Request):
        app_state.reset_counters()
        return {"status": "counters reset"}

    return app
