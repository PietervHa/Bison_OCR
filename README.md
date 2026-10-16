# VisionSoftwareMDE

Machine vision system for automated visual inspection on the MDE Automation production line. The system combines OCR and object detection to assess products (OK/NOK) and communicates the result via a web dashboard and optionally via a TCP connection with the PLC.

---

## Requirements

- Windows 10/11
- Python 3.10+
- Tesseract-OCR installed at `C:/Program Files/Tesseract-OCR/tesseract.exe`
- Webcam or industrial camera (USB)

---

## Installation

```powershell
git clone <repo-url>
cd VisionSoftwareMDE

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and replace the placeholder value before starting the application:

```dotenv
ROBOFLOW_API_KEY=your_key_here
```

Maintenance-mode access is per-user rather than a shared password. Create the first account with:

```powershell
python -m QC_tools.manage_users add <username>
```

You'll be prompted for a password (entered via `getpass`, not shown on screen). See `docs/configuration.md` for the full set of commands (`add`, `passwd`, `list`, `remove`, `logins`).

---

## Starting

```powershell
python -m backend.main
```

The dashboard is available at `http://localhost:5000`.  
Press `Q` in the terminal to manually start an inspection cycle.

---

## Configuration

All settings are in `config/default.yaml`. The most relevant options:

| Setting | Description |
|---|---|
| `vision_mode` | `"ocr"` or `"object_detection"` |
| `object_detection.backend` | `"classifier"`, `"yolo"`, `"template"` or `"roboflow"` |
| `camera.index` | Camera number (0 = first USB camera) |
| `roi` | Region of interest as normalised coordinates (0–1) |
| `trigger.enabled` | `true` to activate TCP connection with PLC |
| `ocr.keywords` | Text that must be recognised for an OK result |

For the Roboflow backend: store the API key as an environment variable (`ROBOFLOW_API_KEY`) and reference it in the config, so the key is not stored in the repository.

For maintenance functions: accounts are managed with `python -m QC_tools.manage_users` (see `docs/configuration.md`), not a config value.

---

## Project structure

```
VisionSoftwareMDE/
├── backend/
│   ├── core/           # Camera, vision engine, AppState, TCP trigger
│   ├── detection/      # OCR and object detection implementations
│   └── output/         # Result storage (JSONL)
├── frontend/           # Web dashboard (Flask + HTML/JS)
├── config/             # Configuration file (YAML)
├── data/
│   ├── dataset/        # Images for classifier training
│   ├── results/        # Inspection results per day (JSONL)
│   ├── templates/      # Reference images for template matching
│   └── logs/           # Log files
├── models/             # Trained models (classifier, YOLO)
├── tools/              # Utility scripts: capture dataset, split, train
├── benchmarks/         # Speed measurement OCR and object detection
├── tests/              # Test scripts
└── docs/               # Extended documentation
```

---

## Training the classifier

```powershell
# 1. Capture images (OK and defective)
python .\tools\capture_dataset.py

# 2. Split dataset (train/val/test)
python .\tools\split_dataset.py

# 3. Train model
python .\tools\train_classifier.py
```

The trained model is saved to `models/classifier/final/`.  
See `docs/classifier-and-object-detection.md` for more information.

---

## PLC connection (TCP)

Enable in `config/default.yaml`:

```yaml
trigger:
  enabled: true
  port: 5001
  trigger_byte: "0x01"
```

The PLC sends byte `0x01` → the system runs an inspection cycle → the PLC receives `OK\n` or `NOK\n`.

---

## Results

Inspection results are stored per day in `data/results/` as JSONL files:

```json
{"timestamp": "2026-06-04T10:30:45", "status": "OK", "confidence": 0.95, "cycle_time_ms": 125}
```

---

## Documentation

| Document | Contents |
|---|---|
| `docs/architecture.md` | System architecture, threading model and data flow |
| `docs/configuration.md` | All configuration options in detail |
| `docs/development.md` | Extending and debugging; debugging instructions |
| `docs/classifier-and-object-detection.md` | Model training and backend selection (classifier, YOLO, Roboflow, template) |
| `docs/project-map.md` | Complete project overview with all modules and functions |
| `docs/roadmap.md` | Future improvements and plans |

---

## Testing

Unit tests and integration tests run with:

```powershell
# All tests
python -m unittest discover -s tests -p "test_*.py"

# Specific test module
python -m unittest tests.test_unit_core
python -m unittest tests.test_integration_vision_pipeline
```

Existing tests:
- `tests/test_unit_core.py` – Unit tests for `AppState` and `InspectionEngine`
- `tests/test_integration_vision_pipeline.py` – Integration tests for the vision pipeline and result storage
- `tests/test_inspection_pipeline.py` – Legacy manual tests with validation

---

*Developed during graduation internship at MDE Automation, February–June 2026.*
