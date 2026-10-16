# Configuration Guide

All settings are managed through `config/default.yaml`. Restart the application after any change.

---

## Quick reference

| Key | Default | Description |
|-----|---------|-------------|
| `vision_mode` | `"ocr"` | `"ocr"` or `"object_detection"` |
| `machine_id` | `"default"` | Identifier used in logs and result files |
| `confidence_threshold` | `0.8` | Min confidence for OK result (0.0–1.0), adjustable at runtime via dashboard |
| `object_detection.backend` | `"classifier"` | `"classifier"`, `"yolo"`, `"template"`, or `"roboflow"` |
| `ocr.engine` | `"tesseract"` | `"tesseract"` or `"paddleocr"` |
| `ocr.preprocess` | `"fast"` | `"fast"` (~50ms) or `"accurate"` (~200ms) |
| `ocr.inference_timeout_s` | `10` | PaddleOCR only: a call slower than this kills and restarts the OCR worker; the cycle is NOK with `failure_reason: ocr_engine_error` |
| `ocr.dynamic_roi.dot_text.enabled` | `true` | OCRead only: remove ribs/mold lines by shape and re-render the print as dark text on white before recognition (replaces `derib_enabled`). Compare variants on real frames with `python -m QC_tools.compare_ocr_preprocessing` |
| `ocr.dynamic_roi.dot_text.bridge_sigma_px` | `0.0` | Merge separate dots into strokes ("slight defocus" done in software). Try 0.5–1.0 on sharp frames |
| `ocr.dynamic_roi.dot_text.illumination` | `"divide"` | `divide` = flat-field by estimated illumination incl. rib shadows (ink in a shadow keeps full contrast); `subtract` = first version |
| `ocr.read_validation.enabled` | `true` | OCRead: report OK only for a read that fits the print template, is confident enough, and (consensus) is confirmed by independent reads. Otherwise NOK with `failure_reason` `format_mismatch` / `low_confidence` / `invalid_value` / `no_consensus` |
| `ocr.read_validation.template` | `["#### ##:##", "BB ##/##"]` | One entry per printed line: `#` digit, `A` letter, `*` any, space = field gap, other characters literal. Product-specific |
| `ocr.read_validation.value_pattern` | hours/minutes/month regex | Regex the canonical code must fully match; `''` disables |
| `ocr.read_validation.min_confidence` | `0.80` | Minimum recognizer confidence for every token used. Calibrate with the QC tool |
| `ocr.read_validation.consensus.variants` | 9 dot-text variants | Preprocessing variants each bottle is read with. OCR-pipeline-style variants (`classic`, `region`, `full_frame`, `ocr_pipeline`) exist but misread print on ribs - check per-variant accuracy with the QC tool first |
| `ocr.read_validation.consensus.fusion` / `min_agree` / `early_stop` | `exact` / `3` / `false` | OK when ≥ `min_agree` reads give the identical code and no other code ties it. `early_stop: true` stops at the first agreeing reads (faster) |
| `ocr.read_validation.consensus.min_position_share` | `0.7` | `position_vote` fusion: weighted share each character's winner needs |
| `ocr.allowed_chars` | letters, digits, `" :/.-"` | PaddleOCR only: tokens without any of these are dropped; other disallowed characters become `?`. `""` disables the filter |
| `camera.index` | `0` | USB camera index; run `tests/camera_test.py` to find yours |
| `camera.flip` | `-1` | `0` = none, `1` = horizontal, `-1` = vertical |
| `trigger.enabled` | `false` | Set to `true` to activate TCP trigger for PLC |
| `trigger.port` | `5001` | Port the PLC connects to |
| `trigger.timeout_s` | `15.0` | Seconds to wait for result before responding NOK |
| `trigger.max_text_length` | `64` | OCRead only: reads longer than this (bytes) are sent as NOK with `failure_reason: text_too_long` |
| `hmi.enable_video_feed` | `true` | Disable to reduce CPU/bandwidth |
| `web.port` | `5000` | Port for the web dashboard |

---

## Region of Interest (ROI)

Coordinates are normalized 0.0–1.0 where `(0,0)` is top-left and `(1,1)` is bottom-right. The ROI box is drawn live on the dashboard — adjust until it tightly covers the inspection target.

```yaml
roi:
  x_start: 0.2
  y_start: 0.7
  x_end: 0.8
  y_end: 0.9
```

```
   0%    20%          80%   100%
0% +-----+------------+-----+
   |     |            |     |
70%|     +============+     |  ← ROI starts here
   |     |  inspect   |     |
90%|     +============+     |  ← ROI ends here
   |     |            |     |
100+-----+------------+-----+
```

---

## OCR settings

Applies when `vision_mode: "ocr"`. The most commonly tuned options:

```yaml
ocr:
  engine: "tesseract"           # or "paddleocr"
  keywords:
    - "MADE IN GERMANY"         # OK if any keyword is found in extracted text
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'  # extracted dates appear in result
  preprocess: "fast"            # "accurate" for difficult images
  psm: 7                        # 7 = single line, 3 = automatic (slower)
  oem: 3                        # 3 = auto, 1 = LSTM only
  whitelist: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
  tesseract_path: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

Switch to PaddleOCR (no installation required) by setting `engine: "paddleocr"`. All other OCR keys are ignored when using PaddleOCR.

---

## Object detection settings

Applies when `vision_mode: "object_detection"`.

```yaml
object_detection:
  backend: "classifier"         # see options below
  classifier:
    model_path: "./models/classifier/final"
  model_path: "models/yolov8n.pt"   # used when backend is "yolo"
  inference_size: 640               # YOLO input resolution
  template:
    references:
      - "data/templates/template_xyz.jpg"
    match_threshold: 0.6
  roboflow:
    model: "cola_detectie"          # profile name to use
    models:
      cola_detectie:
        api_key: "SET_VIA_ENV"      # use ROBOFLOW_API_KEY env variable
        workspace: "your-workspace"
        workflow: "detect-count-and-visualize"
```

The classifier model directory must contain `config.json`, `model.safetensors`, and `preprocessor_config.json`. After retraining, update `classifier.model_path` to the new checkpoint folder.

---

## Secrets

Never commit API keys to Git. Set them via environment variables in a `.env` file (see `.env.example`):

```
ROBOFLOW_API_KEY=your_key_here
```

In `config/default.yaml`, leave the placeholder:

```yaml
roboflow:
  models:
    cola_detectie:
      api_key: "SET_VIA_ENV"
```

---

## Maintenance-mode login

Maintenance mode is gated by per-user accounts stored in the `users` table (same database as inspection results — `database.url` in `config/default.yaml`), not a shared password. There is no registration page in the app on purpose; accounts are managed from the command line with `QC_tools/manage_users.py`, which needs shell access to the machine:

```powershell
# Create a user (prompts for a password via getpass, not shown on screen, not passed as an argument)
python -m QC_tools.manage_users add <username>

# Change a user's password
python -m QC_tools.manage_users passwd <username>

# List all users
python -m QC_tools.manage_users list

# Remove a user (asks you to re-type the username to confirm)
python -m QC_tools.manage_users remove <username>

# Show recent login attempts (successful and failed), newest first
python -m QC_tools.manage_users logins [--limit N]
```

Passwords are hashed with `hashlib.scrypt` (stdlib, no extra dependency) before being stored — see `backend/core/auth.py`. Every login attempt against `/maintenance_mode`, including unknown usernames and wrong passwords, is recorded to the `login_log` table with a timestamp and IP address; the HMI shows the most recent entries in the "Recent Logins" panel while in maintenance mode.

**Unknown usernames** get a distinct message ("No account exists for that username.") rather than the generic wrong-password message. Note this is a deliberate trade-off: it makes typos obvious to a legitimate operator, at the cost of also confirming which usernames don't exist to anyone probing the login form. For an internal, LAN-only industrial HMI this is usually a reasonable trade; reconsider it if the app is ever exposed more broadly.

**Login lockout**: 5 consecutive failed attempts against an existing username, all within a 15-minute span, lock that account out for 15 minutes (`MAX_FAILED_ATTEMPTS` / `FAILURE_WINDOW` / `LOCKOUT_DURATION` in `backend/core/auth.py` — change the constants there to tune it). Only usernames that exist can be locked — an unknown username is rejected before any lockout logic runs, so it can never count as a strike against someone else's account or lock out a real account by mistake. A successful login resets the streak. Continuing to try a locked account extends the lock. There's no separate "locked" flag anywhere — lockout state is always derived from `login_log` at request time, so it can't drift out of sync.

The "Clear" button on the Recent Logins panel (or `DELETE /login_log`, maintenance-gated) wipes the whole log. Since lockout state is derived from that same table, clearing it also immediately lifts any lockout currently in effect — that's intentional, and doubles as a manual "unlock this account" action for an admin.

Maintenance sessions are single-active: logging in from a second device silently takes over the session, same as before this change — there's still just one operator "in" maintenance mode at a time, it's now just tied to a specific username instead of a shared password.

---

## Startup validation errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model_path does not exist` | Path to classifier or YOLO model not found | Check path is relative to repo root and model has been trained |
| `invalid backend` | Unknown backend name | Valid values: `"classifier"`, `"yolo"`, `"template"`, `"roboflow"` |
| `classifier profile not found` | `roboflow.model` not in `roboflow.models` | Add the profile or correct the name |
| `ROI coordinates invalid` | ROI values outside 0–1 | All four ROI values must be between 0.0 and 1.0 |

---

## Complete configuration examples

### OCR — reading a product label

```yaml
vision_mode: "ocr"
confidence_threshold: 0.85

roi:
  x_start: 0.1
  y_start: 0.3
  x_end: 0.9
  y_end: 0.7

camera:
  index: 0
  width: 1920
  height: 1440
  flip: -1

ocr:
  engine: "tesseract"
  keywords:
    - "MADE IN GERMANY"
    - "CE"
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'
  preprocess: "accurate"
  psm: 3
```

### Local classifier — OK/defective detection

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.75

object_detection:
  backend: "classifier"
  classifier:
    model_path: "models/classifier/final"

camera:
  index: 0
  width: 1280
  height: 1024
  flip: -1
```

### YOLO with PLC trigger

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.6

object_detection:
  backend: "yolo"
  model_path: "models/yolov8n.pt"
  inference_size: 640

trigger:
  enabled: true
  host: "0.0.0.0"
  port: 5001
  trigger_byte: "0x01"
  response_ok: "OK\n"
  response_nok: "NOK\n"
  timeout_s: 3.0
```

### Roboflow cloud inference

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.7

object_detection:
  backend: "roboflow"
  roboflow:
    model: "cola_detectie"
    models:
      cola_detectie:
        api_key: "SET_VIA_ENV"
        workspace: "pieters-workspace-kugm8"
        workflow: "detect-count-and-visualize"
        api_url: "https://serverless.roboflow.com"
```