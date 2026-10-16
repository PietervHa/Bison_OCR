# Architecture

## Components

```
            ┌──────────────── Web HMI (FastAPI + HTML/JS, port 5000) ───────────────┐
            │  /  /status  /result  /video_feed  /camera_status                     │
            │  /maintenance_mode  /camera_rotation  /reset_counters                 │
            └───────────────────────────────┬───────────────────────────────────────┘
                                            │
                              AppState (lock-protected)
                     last result · counters · maintenance session · rotation
                                            │
     ┌──────────────────┬───────────────────┼─────────────────────┐
     ▼                  ▼                   ▼                     ▼
  Camera           Q-key loop        TCP trigger server     result_writer
 (capture +        (main thread)     (PLC, port 5001)       data/results/<date>.jsonl
  reconnect)             │                   │
                         └──── run_vision ───┘
                               (backend/core/vision.py, 1 worker thread)
                                        │
                               OCRead.read(frame)
                         (backend/detection/ocread/engine.py)
```

## Threads

| Thread | Role |
|---|---|
| Main | `vision_trigger_loop`: waits for the `Q` key and starts a read |
| Web server | uvicorn serving `frontend/web.py` |
| Vision worker | single-thread pool in `vision.py`: runs `OCRead.read` and calls back with the normalised result |
| TCP trigger | accepts the PLC connection, reads trigger bytes, waits for the result, answers |
| Camera | capture worker and supervisor (stall/frozen-frame detection, reconnect by device name) |

`AppState` guards all shared state with a lock. The Q-key loop uses a `threading.Event` so key presses don't overlap reads.

## One read

1. **Trigger**: the PLC sends the trigger byte, or someone presses `Q`.
2. **Frame**: the latest frame from the camera. If there is no frame, the TCP server records `camera_unavailable` and answers `NOK`.
3. **Locate** (`text_locator.locate_text_region`): blackhat ink map, long rib lines erased, components clustered into the text block. Returns a rotated rect, or `None` (then the full frame is read).
4. **Crop** (`text_locator.deskew_crop`): straightened crop with 20 px extra margin. Then `ocr.downscale`/`min_dim` (no-op for today's crop size).
5. **Font reader** (`font_reader.FontReader.read`): `dot_text.clean_dot_text` cleanup, grid segmentation, classifier per character cell, both orientations tried. Returns lines, text, `literal_score` and the lowest digit confidence, or `None`.
6. **Validate**:
   - Reject if `literal_score < min_literal_score` (`format_mismatch`).
   - Reject if digit confidence `< min_digit_confidence` (`low_confidence`).
   - Otherwise the template check (`CodeTemplate.validate`: format and `value_pattern`) decides.
7. **Result**: built by the engine and normalised in `vision._normalize_result`. A rejected read always has `text == ""` and `detections == []`.
8. **Record**: `main._process_vision_result` adds `cycle_time_ms`, updates AppState and counters, and appends to the JSONL file.
9. **Answer**: the TCP server sends the code text plus `\n` on OK, otherwise `NOK\n`.

With `ocr.dynamic_roi.debug_save: true`, each read writes a folder `data/debug/dynamic_roi/<timestamp>/` containing:

- `text_locator_*.png`
- `crop_raw_preprocess.png`
- `font_reader.txt`, the read summary, including the rejected decode, which is never shown elsewhere

## Result contract

```json
{
  "mode": "ocread",
  "status": "OK",
  "text": "4113 08:26 BB 05/26",
  "detections": [{"text": "4113 08:26", "confidence": 0.99}, {"text": "BB 05/26", "confidence": 0.99}],
  "confidence": 0.99,
  "validation": {"accepted": true, "reason": null, "fusion": null,
                 "reads": [{"variant": "font_reader", "accepted": true, "reason": null,
                            "text": "4113 08:26 BB 05/26", "raw_text": "4113 08:26 BB 05/26",
                            "min_token_confidence": 0.99, "literal_score": -0.2,
                            "timings_ms": {}, "orientation": "normal"}]},
  "processing_time_ms": 412.3,
  "cycle_time_ms": 420.1,
  "timestamp": "2026-10-13T09:43:03.123456"
}
```

A NOK result has `"status": "NOK"` and a `failure_reason`. A technical failure also has `"error"`.

## Status rules (`vision._normalize_result`)

| Condition | Status | `failure_reason` |
|---|---|---|
| `error` set (exception, missing frame, font reader unavailable) | NOK | `ocr_engine_error` |
| `validation.accepted` is false | NOK | the validation reason |
| no text and no detections | NOK | `no_text_detected` |
| otherwise | OK | – |

## Failure reasons

| Reason | Meaning |
|---|---|
| `no_text_detected` | no print found on the frame and nothing readable |
| `format_mismatch` | print found, but the read doesn't fit the template (or `literal_score` too low) |
| `low_confidence` | a digit below `min_digit_confidence` |
| `invalid_value` | fits the format but fails `value_pattern` (e.g. hour 27, month 13) |
| `ocr_engine_error` | the read itself failed: exception, no frame (Q-key path), or font reader unavailable |
| `text_too_long` | OK read longer than `trigger.max_text_length` bytes; sent as NOK |
| `camera_unavailable` | TCP trigger, but the camera had no frame |
| `vision_start_failed` | TCP trigger, but the read could not be started |
| `vision_timeout` | TCP trigger, no result within `trigger.timeout_s` |

## Maintenance mode

`POST /maintenance_mode` with `{"maintenance_mode": true, "password": "..."}`:

- `MAINTENANCE_PASSWORD` (from `.env` or the environment) not set or empty: 403, maintenance stays locked.
- Wrong password: 403 after about 1 s. The comparison uses `secrets.compare_digest`.
- Right password: maintenance on, and an httponly session cookie `maintenance_session` (30 min). `/camera_rotation` and `/reset_counters` require that cookie.

`{"maintenance_mode": false}` leaves maintenance mode and clears the session.
