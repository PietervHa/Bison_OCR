import anyio
import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
import time

log = get_logger(__name__)


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


    # 15 fps is plenty for the HMI preview and halves the encode work of 30.
    STREAM_FPS = float(cfg.get("hmi", {}).get("stream_fps", 15))
    FRAME_INTERVAL_S = 1.0 / STREAM_FPS
    open_streams = [0]  # mutated only on the event loop thread

    def _encode(frame):
        # encode with configured quality instead of OpenCV default (95)
        _, buffer = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    def _grab_and_encode():
        frame = camera.get_frame()
        return None if frame is None else _encode(frame)

    async def generate_frames(client):
        # Async generator: on client disconnect Starlette cancels the response
        # task, which raises CancelledError at one of the awaits below, so the
        # stream ends right away. The sleeps run on the event loop instead of
        # holding a threadpool worker; only the grab+encode uses a thread.
        open_streams[0] += 1
        log.info("video_feed opened (%s), open streams: %d", client, open_streams[0])
        try:
            if not cfg["hmi"]["enable_video_feed"]:
                import numpy as np
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Video feed disabled", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # quality applies to the disabled-feed placeholder too
                chunk = _encode(blank)
                while True:
                    yield chunk
                    await anyio.sleep(1)

            while True:
                frame_start = time.perf_counter()
                chunk = await anyio.to_thread.run_sync(_grab_and_encode)
                if chunk is None:
                    await anyio.sleep(0.01)
                    continue
                yield chunk
                remaining = FRAME_INTERVAL_S - (time.perf_counter() - frame_start)
                if remaining > 0:
                    await anyio.sleep(remaining)
        finally:
            open_streams[0] -= 1
            log.info("video_feed closed (%s), open streams: %d", client, open_streams[0])

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
    def video_feed(request: Request):
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        return StreamingResponse(
            generate_frames(client),
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
