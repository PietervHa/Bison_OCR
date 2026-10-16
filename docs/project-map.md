# Project map

This document gives a high-level view of the repository structure and what each area is used for.

## Repository structure

```text
VisionSoftwareMDE/
├── README.md
├── requirements.txt
├── ocr_debug.py
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── camera.py
│   │   ├── config_loader.py
│   │   ├── inspection_engine.py
│   │   ├── state.py
│   │   ├── tcp_trigger_server.py
│   │   └── vision.py
│   ├── detection/
│   │   ├── objectdetection/
│   │   └── ocr/
│   ├── models/
│   │   └── yolov8n.pt
│   ├── output/
│   │   └── result_writer.py
│   └── utils/
│       ├── logger.py
│       └── roi.py
├── benchmarks/
│   ├── hmi_polling_benchmark.py
│   ├── object_detection_benchmark.py
│   └── ocr_benchmark.py
├── config/
│   └── default.yaml
├── data/
│   ├── dataset/
│   ├── dataset_split/
│   ├── logs/
│   ├── results/
│   └── templates/
├── debug_ocr/
│   ├── 00_raw_frame.png
│   ├── 01_after_roi.png
│   ├── 02_channel_*.png
│   ├── 03_gray_*.png
│   ├── 04_after_upscale.png
│   └── 05_*_threshold.png
├── docs/
│   └── classifier-and-object-detection.md
├── frontend/
│   ├── web.py
│   ├── static/
│   └── templates/
├── models/
│   ├── cola_detectie.py
│   ├── yolov8n.pt
│   └── classifier/
├── tests/
│   ├── camera_test.py
│   ├── paddleocr_test.py
│   ├── paddleocr_test_core.py
│   ├── run_ocr_test.py
│   ├── test_inspection_pipeline.py
│   ├── images/
│   ├── models/
│   └── results/
└── tools/
    ├── capture_dataset.py
    ├── capture_template.py
    ├── split_dataset.py
    └── train_classifier.py
```

## What each folder does

### Root files
- `README.md` — short project overview.
- `requirements.txt` — Python dependencies for the project.
- `ocr_debug.py` — OCR debugging script for analyzing and visualizing preprocessing pipeline.

### `backend/`
Main application logic for the vision system.

- `backend/main.py` — backend entry point.
- `backend/core/` — core runtime services such as camera access, state, configuration, inspection flow, and trigger handling.
- `backend/detection/` — OCR and object-detection implementations.
- `backend/models/` — backend-side bundled models or model assets.
- `backend/output/` — result formatting and writing.
- `backend/utils/` — shared helpers like logging and ROI utilities.

### `benchmarks/`
Scripts used to measure performance of OCR, object detection, and HMI polling.

### `config/`
Application configuration, especially `default.yaml`.

### `data/`
Working data used by the project.

- `dataset/` — captured raw images.
- `dataset_split/` — train/validation/test split used for training.
- `logs/` — runtime or experiment logs.
- `results/` — exported results and outputs.
- `templates/` — image templates or reference assets.

### `debug_ocr/`
OCR preprocessing debug outputs including intermediate image processing stages.

- Raw frame captures and post-ROI processing
- Channel-wise analysis (R, G, B channels)
- Grayscale conversions (luminance, best channel)
- Upscaled images
- Threshold comparisons (Otsu, adaptive)

### `docs/`
Project documentation.

- `classifier-and-object-detection.md` — guide for classifier training and detector backend selection.
- `project-map.md` — this repository structure overview.

### `frontend/`
Web UI or API-facing frontend layer.

- `frontend/web.py` — frontend web application entry point.
- `frontend/static/` — static assets.
- `frontend/templates/` — HTML templates.

### `models/`
Shared model files and model-related scripts.

- `cola_detectie.py` — Roboflow / detection profile helper.
- `yolov8n.pt` — YOLO model checkpoint.
- `classifier/` — trained classifier output directory.

### `tests/`
Automated tests and test helpers for camera, OCR, and inspection pipeline behavior.

- `camera_test.py`, `paddleocr_test.py`, `run_ocr_test.py`, `test_inspection_pipeline.py` — test modules.
- `images/` — test image assets and fixtures.
- `models/` — test model files.
- `results/` — test output and results.

### `tools/`
Utility scripts for dataset capture, dataset splitting, template capture, and classifier training.

## Notes

- Some folders may contain generated or environment-specific files that are intentionally omitted from the tree above.
- Large dataset folders may have additional nested files not listed here.
- The structure is organized around three main concerns: vision runtime, model training/evaluation, and supporting tools/documentation.

