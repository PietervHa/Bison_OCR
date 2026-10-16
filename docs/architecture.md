# Architecture Guide

---

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Dashboard                         │
│                    (Flask + HTML/JavaScript)                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/JSON
┌─────────────────────▼────────────────────────────────────────┐
│                    Frontend Layer                             │
│  (web.py — /video_feed, /status, /threshold endpoints)      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Application State (AppState)               │
│              shared across threads, lock-protected           │
└────────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┼──────────────┐
    ▼           ▼              ▼
┌──────────┐ ┌─────────────┐ ┌──────────────────────┐
│  Camera  │ │Vision Engine│ │ TCP Trigger Server   │
│          │ │             │ │ (optional, port 5001)│
│ captures │ │ • OCR       │ │                      │
│ frames   │ │ • Detection │ │ PLC → trigger byte   │
│ in bg    │ │ • Inference │ │ server → OK/NOK      │
└──────────┘ └─────────────┘ └──────────────────────┘
```

---

## Threading model

The application runs five concurrent threads:

| Thread | Role |
|--------|------|
| Main | Vision trigger loop — waits for Q key or TCP trigger, calls `run_vision()` |
| Web server | Flask on port 5000 — serves dashboard, video stream, API |
| Vision workers | ThreadPoolExecutor — runs inference in background, fires callback when done |
| TCP trigger (optional) | Listens for PLC connections on port 5001 |
| Camera | Continuously captures frames into a single-frame buffer |

`AppState` uses `threading.Lock` for all shared state. A `threading.Event` flag prevents overlapping vision cycles.

---

## Vision cycle — data flow

```
1. TRIGGER (Q key or TCP byte from PLC)
   ↓
2. CAPTURE latest frame from camera buffer
   ↓
3. VISION WORKER (background thread)
   ├─ Extract ROI from frame
   ├─ Preprocess image
   ├─ Run inference (OCR or object detection)
   └─ InspectionEngine → OK / NOK / ERROR
   ↓
4. RESULT CALLBACK
   ├─ Update AppState (counters, last result)
   ├─ Save JSONL to data/results/
   └─ Log to data/logs/
   ↓
5. Dashboard updates / TCP client receives OK or NOK

Typical cycle time: 50–300 ms depending on model and hardware.
```

---

## Key modules

### `backend/core/state.py` — `AppState`

Single shared object holding all runtime state: vision mode, confidence threshold, OK/NOK/error counters, last result, classifier load status. All reads and writes go through a threading lock.

### `backend/core/vision.py`

Orchestrates the vision pipeline. Routes to OCR or object detection based on config, manages model loading, dispatches work to the ThreadPoolExecutor, and fires the result callback.

Key functions:
- `bind_app_state(app_state)` — called once at startup
- `run_vision(frame, callback)` — triggers async vision cycle

### `backend/core/inspection_engine.py` — `InspectionEngine`

Translates raw detection output (confidence, labels, boxes) into a final OK / NOK / ERROR verdict by comparing against the active confidence threshold.

### `backend/core/tcp_trigger_server.py` — `TCPTriggerServer`

Listens on the configured TCP port. On receipt of the trigger byte, calls `run_vision()`, waits for the result (up to `timeout_s`), and sends `OK\n` or `NOK\n` back to the PLC.

### `backend/detection/ocr/`

Two interchangeable engines — Tesseract and PaddleOCR — behind a common interface. Both extract ROI, preprocess the image, run OCR, then match against keywords and date regex. Debug frames are written to `debug_ocr/`.

### `backend/detection/objectdetection/`

Three interchangeable backends behind a common interface:

| Backend | How it works |
|---------|-------------|
| `classifier` | Local HuggingFace image-classification pipeline (PyTorch) |
| `yolo` | Local Ultralytics YOLO `.pt` model |
| `template` | OpenCV template matching against reference images |
| `roboflow` | REST call to Roboflow hosted inference workflow |

### `backend/output/result_writer.py`

Appends each inspection result as a JSON line to `data/results/YYYY-MM-DD.jsonl`. Result format:

```json
{
  "timestamp": "2026-06-04T10:30:45.123456",
  "vision_mode": "object_detection",
  "status": "OK",
  "confidence": 0.95,
  "confidence_threshold": 0.8,
  "cycle_time_ms": 125.4,
  "detections": [{"class": "ok", "confidence": 0.98, "box": [100, 200, 350, 450]}]
}
```

---

## State diagram

```
[IDLE]
  │
  ├─ Q key / TCP trigger
  ▼
[CAPTURING FRAME]
  ▼
[VISION_BUSY — flag set]
  │  background thread: preprocess → inference → inspection engine
  ▼
[RESULT CALLBACK — update state, save, log]
  ▼
[VISION_BUSY — flag cleared]
  ▼
[IDLE — ready for next trigger]
```

---

## Performance reference

| Step | Typical time share |
|------|--------------------|
| Inference (classifier / YOLO) | 60–80% of cycle (50–200 ms) |
| OCR preprocessing | 10–20% of cycle (50–200 ms) |
| Frame capture | 5–10% of cycle |

To reduce cycle time: shrink the ROI, switch to a smaller model (`yolov8n` vs `yolov8m`), set `ocr.preprocess: "fast"`, or enable GPU (see Roadmap).

---

## Error handling

| Error type | Behaviour |
|------------|-----------|
| Vision processing exception | Caught in trigger loop, logged with stack trace, result set to `ERROR`, flag cleared, app continues |
| Configuration error at startup | Logged and app exits with message |
| Camera failure | Logged, frame returns `None`, vision cycle skipped |

---

## Extending the system

### Add a new detection backend

1. Create a class in `backend/detection/objectdetection/` with a `detect(frame, roi_coords)` method that returns `{status, confidence, detections}`.
2. Register it in `backend/core/vision.py` — add a branch in the backend dispatcher.
3. Add a config section in `config/default.yaml`.

### Add a new OCR engine

1. Implement in `backend/detection/ocr/` matching the interface of `tesseract_ocr.py`.
2. Add a branch in the OCR dispatcher.
3. Add any engine-specific config keys under `ocr:` in `default.yaml`.

### Change result storage

Replace or extend `backend/output/result_writer.py`. The callback in `vision.py` passes the full result dict — redirect it to a database, webhook, or any other sink without touching the rest of the pipeline.