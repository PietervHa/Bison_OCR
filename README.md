# Bison OCR – OCRead

Reads the dot-printed code on bottle caps (e.g. `4113 08:26` / `BB 05/26`, `6274 05:58` / `EXP 05/28`, or the single line `6286 16:24`) on the MDE Automation production line. A camera frame goes through a classical-CV text locator and a classifier trained on the printer's dot font (the "font reader"). The read is then checked against the print format. The result goes to the PLC over TCP and is shown on a web HMI.

A read is either a validated code or an explicit NOK with a reason. A rejected read never shows or sends a guessed code.

---

## Requirements

- Windows 10/11
- Python 3.10+
- USB camera (configured for a DFK 22BUC03; set `camera.device_name`)

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and set the maintenance password:

```dotenv
MAINTENANCE_PASSWORD=choose-a-password
```

If `MAINTENANCE_PASSWORD` is not set (or empty), maintenance mode stays locked.

---

## Starting

```powershell
python -m backend.main
# or with another config file:
python -m backend.main --config path\to\config.yaml
```

- HMI: `http://localhost:5000`
- PLC trigger server: TCP port `5001` (when `trigger.enabled: true`)
- Press `Q` (global keyboard hook) to trigger a read by hand.

---

## Pipeline

```
camera frame
  -> text locator (backend/detection/ocr/text_locator.py)   finds the print, strips rib lines
  -> deskewed crop + optional downscale
  -> font reader (backend/detection/ocr/font_reader.py)     dot_text.py cleanup + trained classifier
     (once per configured print format; the trusted format with the most lines wins)
  -> format check (backend/detection/ocr/code_validator.py) template + value_pattern
  -> result (mode "ocread")
```

The orchestration is `OCRead.read(frame)` in `backend/detection/ocread/engine.py`. `backend/core/vision.py` runs it in a worker thread and decides OK/NOK.

---

## PLC connection (TCP)

```yaml
trigger:
  enabled: true
  port: 5001
  trigger_byte: "0x01"
  timeout_s: 15.0
  max_text_length: 64
```

The PLC sends the trigger byte and the system runs one read. On OK the PLC receives the code text plus a newline, e.g. `4113 08:26 BB 05/26\n`. On anything else it receives `NOK\n`: a rejected read, no camera frame, a timeout, or a read longer than `max_text_length`. The PLC compares the code with the expected value itself.

---

## Results

Every read is appended to `data/results/<date>.jsonl`, for example:

```json
{"text": "4113 08:26 BB 05/26", "status": "OK", "mode": "ocread", "confidence": 0.99,
 "detections": [...], "validation": {...}, "processing_time_ms": 412.3, "cycle_time_ms": 420.1,
 "timestamp": "2026-10-13T09:43:03.123456"}
```

A NOK result also has `failure_reason` (see `docs/architecture.md`). Logs are written to `data/logs/vision.log`.

---

## QC tools

```powershell
python -m QC_tools.q_trigger --max-triggers 100        # send TCP triggers like the PLC
python -m QC_tools.ocread_session_report --expected "4113 08:26 BB 05/26"   # report from today's JSONL
python -m QC_tools.camera_stability --seconds 30 --label locked              # exposure stability check
```

---

## Testing

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m pyflakes backend frontend QC_tools tests
```

The maintenance-password tests use FastAPI's `TestClient`. They are skipped when `httpx` is not installed (`pip install httpx` to run them).

With `camera.source: "images"` the app reads still images from `camera.images_dir`; images wider than `camera.images_max_width` (default 744, the camera's width) are scaled down to it so the OCRead pipeline sees camera-sized characters.

---

## Documentation

| Document | Contents |
|---|---|
| `docs/architecture.md` | Components, threads, data flow, result contract, failure reasons |
| `docs/configuration.md` | Every configuration key |
| `docs/development.md` | Tests, QC tools, debug captures, changing the print formats |
| `docs/project-map.md` | Files and what they do |
| `docs/roadmap.md` | Open items |
| `docs/VisionLinkTest.st` | PLC-side test program (Structured Text) |
