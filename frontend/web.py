import csv
import io
import json
import secrets
from datetime import date, datetime, timedelta
import cv2
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openpyxl import Workbook
from pydantic import BaseModel
from pathlib import Path
from backend.core import auth, db
from backend.core.config_loader import cfg
from backend.utils.roi import draw_roi
from QC_tools.capture_dataset import DatasetCapture
import time


def _resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return resolved.resolve()


# Only paths under this directory may be loaded as a classifier, even by a
# maintenance-authenticated request. Prevents /load_classifier from being
# pointed at an arbitrary location on disk.
_MODELS_ROOT = (Path(__file__).resolve().parents[1] / "models").resolve()


def _is_within_models_dir(resolved_path: Path) -> bool:
    try:
        return resolved_path.is_relative_to(_MODELS_ROOT)
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        try:
            resolved_path.relative_to(_MODELS_ROOT)
            return True
        except ValueError:
            return False


# Columns that always appear first, in this order, in an export.
_CORE_EXPORT_COLUMNS = [
    "timestamp", "status", "mode", "confidence_threshold",
    "processing_time_ms", "cycle_time_ms", "error",
]


def _build_export_table(results: list) -> tuple:
    extra_columns = set()
    for r in results:
        extra_columns.update(k for k in r.keys() if k not in _CORE_EXPORT_COLUMNS)
    headers = _CORE_EXPORT_COLUMNS + sorted(extra_columns)

    rows = []
    for r in results:
        row = []
        for col in headers:
            value = r.get(col)
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            row.append(value)
        rows.append(row)
    return headers, rows


# --- Request bodies -------------------------------------------------------
# Pydantic models replace Flask's `request.json or {}` pattern. Defaults
# mirror the old `data.get(key, default)` calls exactly.

class ThresholdBody(BaseModel):
    threshold: float = 0.5


class MaintenanceModeBody(BaseModel):
    maintenance_mode: bool = False
    username: str = ""
    password: str = ""


class VisionModeBody(BaseModel):
    vision_mode: str = ""


class OcrKeywordBody(BaseModel):
    ocr_keyword: str = ""


class LoadClassifierBody(BaseModel):
    model_path: str = ""


class DatasetCaptureBody(BaseModel):
    label: str = ""


def create_app(camera, app_state) -> FastAPI:
    app = FastAPI()

    db.init_db()

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

    # shared DatasetCapture instance for the lifetime of this app
    dataset_capture = DatasetCapture()

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

            frame_for_stream = frame
            roi = cfg.get("roi")
            # OCRead has no static ROI at all (see paddle_ocr.py _apply_roi) -
            # it locates and deskews the text region fresh on every frame, so
            # a fixed box here would be meaningless and wouldn't track what's
            # actually being read. Only draw it for the plain "ocr" mode,
            # which still uses the static box.
            if cfg["hmi"]["debug_draw_roi"] and app_state.get_vision_mode() == "ocr" and roi:
                # use shared draw_roi instead of inline _draw_roi helper
                frame_for_stream = draw_roi(frame.copy(), roi)

            # encode with configured quality instead of OpenCV default (95)
            _, buffer = cv2.imencode(".jpg", frame_for_stream,
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
            "vision_mode": app_state.get_vision_mode(),
            "maintenance_mode": is_maintenance,
            "username": app_state.get_maintenance_session_user() if is_maintenance else None,
            "machine_id": cfg["machine_id"],
            "version": "1.0.0",
        }

    @app.get("/result")
    def get_result():
        return app_state.get_snapshot()

    @app.get("/threshold")
    def get_threshold():
        return {"threshold": app_state.get_threshold()}

    @app.post("/threshold")
    def set_threshold(body: ThresholdBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.set_threshold(body.threshold)
        return {"threshold": app_state.get_threshold()}

    @app.post("/maintenance_mode")
    def set_maintenance_mode(body: MaintenanceModeBody, request: Request, response: Response):
        # Entering maintenance mode always requires a valid username+password
        # against the `users` table. Every attempt -- success or failure,
        # including unknown usernames -- is logged to login_log. Unknown
        # usernames are called out explicitly (rather than folded into a
        # generic "incorrect" message) and, since there's no real account to
        # attach a strike to, they can never trigger a lockout -- only a
        # streak of failures against a username that actually exists can.
        if body.maintenance_mode:
            username = body.username.strip()
            client_ip = request.client.host if request.client else None

            user = db.get_user(username) if username else None
            if user is None:
                # Still run a scrypt hash against a dummy value so an unknown
                # username takes roughly as long as a known one with a wrong
                # password -- a minor defense-in-depth measure even though
                # the message below already tells the caller the account
                # doesn't exist.
                auth.verify_password(body.password, auth.DUMMY_PASSWORD_HASH)
                db.record_login(username or "(empty)", False, client_ip)
                return JSONResponse(
                    status_code=403,
                    content={"error": "unknown_user", "message": "No account exists for that username."},
                )

            recent_failures = db.get_recent_failures_since_last_success(username, limit=auth.MAX_FAILED_ATTEMPTS)
            unlock_at = auth.lockout_until(recent_failures)
            if unlock_at is not None:
                # Locked: don't even check the password, and log this attempt
                # as a failure too, so continuing to hammer a locked account
                # extends the lock rather than doing nothing.
                db.record_login(username, False, client_ip)
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "locked_out",
                        "message": f"Account locked after repeated failed attempts. Try again after {unlock_at.strftime('%H:%M:%S')}.",
                        "locked_until": unlock_at.isoformat(),
                    },
                )

            password_ok = auth.verify_password(body.password, user["password_hash"])
            db.record_login(username, password_ok, client_ip)

            if not password_ok:
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid_password", "message": "Incorrect password."},
                )

            session_token = secrets.token_urlsafe(24)
            app_state.set_maintenance_mode(True)
            app_state.set_maintenance_session_token(session_token)
            app_state.set_maintenance_session_user(username)
            response.set_cookie(
                "maintenance_session",
                session_token,
                httponly=True,
                samesite="lax",
                max_age=60 * 30,
                path="/",
            )
            return {"maintenance_mode": True, "username": username}

        app_state.set_maintenance_mode(False)
        app_state.set_maintenance_session_token("")
        app_state.set_maintenance_session_user("")
        response.delete_cookie("maintenance_session", path="/")
        return {"maintenance_mode": False}

    @app.delete("/login_log")
    def clear_login_log(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        deleted = db.clear_login_log()
        return {"cleared": deleted}

    @app.get("/login_log")
    def login_log(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        return {"logins": db.get_recent_logins(limit=20)}

    @app.post("/vision_mode")
    def set_vision_mode(body: VisionModeBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.set_vision_mode(body.vision_mode)
        return {"vision_mode": app_state.get_vision_mode()}

    @app.post("/camera_rotation")
    def rotate_camera(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.rotate_camera()
        return {"camera_rotation": app_state.get_camera_rotation()}

    @app.get("/ocr_keyword")
    def get_ocr_keyword():
        return {"ocr_keyword": app_state.get_ocr_keyword()}

    @app.post("/ocr_keyword")
    def set_ocr_keyword(body: OcrKeywordBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        new_keyword = body.ocr_keyword.strip()
        if not new_keyword:
            return JSONResponse(status_code=400, content={"error": "ocr_keyword cannot be empty"})
        app_state.set_ocr_keyword(new_keyword)
        return {"ocr_keyword": app_state.get_ocr_keyword()}

    @app.post("/load_classifier")
    def load_classifier(body: LoadClassifierBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        model_path = body.model_path.strip()
        resolved_path = _resolve_repo_path(model_path)
        if not model_path or not resolved_path.exists():
            return JSONResponse(status_code=400, content={"error": "model_path does not exist"})
        if not _is_within_models_dir(resolved_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"model_path must be inside {_MODELS_ROOT.name}/"},
            )
        classifier_loaded = bool(app_state.load_classifier(str(resolved_path)))
        return {"classifier_loaded": classifier_loaded, "model_path": str(resolved_path)}

    @app.get("/classifier_status")
    def get_classifier_status():
        status = app_state.get_classifier_status()
        return {
            "classifier_loaded": bool(status.get("loaded", False)),
            "model_path": status.get("model_path") or None,
        }

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

    @app.post("/dataset/capture")
    def dataset_capture_route(body: DatasetCaptureBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False, "error": "Not in maintenance mode"})
        label = body.label.lower()
        if label not in ("ok", "defective"):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "label must be 'ok' or 'defective'"},
            )

        frame = camera.get_frame()
        if frame is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "no frame available"},
            )

        # apply the same rotation logic as DatasetCapture
        rotated = dataset_capture._apply_rotation(frame)

        try:
            dataset_capture._save_frame(rotated, label)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

        return {
            "success": True,
            "ok": dataset_capture.counters.get("ok", 0),
            "defective": dataset_capture.counters.get("defective", 0),
        }

    @app.get("/dataset/counts")
    def dataset_counts():
        return dataset_capture.counters

    @app.get("/analytics")
    def analytics():
        template_path = Path(__file__).resolve().parent / "templates" / "analytics.html"
        return FileResponse(template_path)

    @app.get("/analytics/data")
    def analytics_data(date_param: str = Query(default="", alias="date")):
        try:
            # Parse requested date, default to today, clamp to 7-day window
            today = date.today()
            min_date = today - timedelta(days=6)  # 7 days including today

            raw_date = date_param
            try:
                requested_date = date.fromisoformat(raw_date) if raw_date else today
            except ValueError:
                requested_date = today

            # Clamp: never go beyond today or before 7 days ago
            if requested_date > today:
                requested_date = today
            if requested_date < min_date:
                requested_date = min_date

            results = db.query_summary(requested_date, requested_date)

            total = len(results)
            ok_count = sum(1 for r in results if r.get("status") == "OK")
            nok_count = total - ok_count

            processing_times = [
                r["processing_time_ms"] for r in results
                if isinstance(r.get("processing_time_ms"), (int, float))
            ]
            avg_processing_ms = round(
                sum(processing_times) / len(processing_times), 2
            ) if processing_times else 0.0

            # Build timeline: group results by hour (0-23), count OK and NOK per hour
            timeline = {str(h): {"ok": 0, "nok": 0} for h in range(24)}
            for r in results:
                ts = r.get("timestamp")
                if not isinstance(ts, datetime):
                    continue
                hour = str(ts.hour)
                if r.get("status") == "OK":
                    timeline[hour]["ok"] += 1
                else:
                    timeline[hour]["nok"] += 1

            # Builds speed timeline in 15-minute buckets
            # Key format: "HH:MM" for each 15-min slot (00:00, 00:15, 00:30, 00:45, 01:00 ...)
            speed_buckets = {}
            for r in results:
                ts = r.get("timestamp")
                pt = r.get("processing_time_ms")
                if not isinstance(pt, (int, float)) or not isinstance(ts, datetime):
                    continue
                minute_slot = (ts.minute // 15) * 15
                key = f"{ts.hour:02d}:{minute_slot:02d}"
                speed_buckets.setdefault(key, []).append(float(pt))

            speed_timeline = {}
            for key, times in speed_buckets.items():
                speed_timeline[key] = {
                    "avg": round(sum(times) / len(times), 2),
                    "min": round(min(times), 2),
                    "max": round(max(times), 2),
                }

            return {
                "date": requested_date.isoformat(),
                "is_today": requested_date == today,
                "is_min_date": requested_date <= min_date,
                "total": total,
                "ok_count": ok_count,
                "nok_count": nok_count,
                "ok_rate": round(ok_count / total * 100, 1) if total > 0 else 0.0,
                "avg_processing_ms": avg_processing_ms,
                "timeline": timeline,
                "speed_timeline": speed_timeline,
            }

        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/analytics/export")
    def analytics_export(
        start: str = Query(default=""),
        end: str = Query(default=""),
        export_format: str = Query(default="csv", alias="format"),
    ):
        try:
            try:
                start_date = date.fromisoformat(start)
                end_date = date.fromisoformat(end)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "start and end must be dates in YYYY-MM-DD format"},
                )

            if start_date > end_date:
                start_date, end_date = end_date, start_date

            if export_format not in ("csv", "xlsx"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "format must be 'csv' or 'xlsx'"},
                )

            results = db.query_full(start_date, end_date)
            headers, rows = _build_export_table(results)
            filename_base = f"inspections_{start_date.isoformat()}_to_{end_date.isoformat()}"

            if export_format == "csv":
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerow(headers)
                writer.writerows(rows)
                # utf-8-sig so Excel recognizes the encoding instead of mangling accented characters
                csv_bytes = buffer.getvalue().encode("utf-8-sig")
                return Response(
                    content=csv_bytes,
                    media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename_base}.csv"'},
                )

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Inspections"
            sheet.append(headers)
            for row in rows:
                sheet.append(row)
            xlsx_buffer = io.BytesIO()
            workbook.save(xlsx_buffer)
            return Response(
                content=xlsx_buffer.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{filename_base}.xlsx"'},
            )

        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app