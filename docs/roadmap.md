# Roadmap – VisionSoftwareMDE

This document describes known improvements and possible extensions. Items are divided into **to be resolved now** (minor bugs) and **future development** (features planned after delivery).

---

## To be resolved now (directly after delivery)

### Bug – Error logging on camera failure (FTC 45)
**Problem:** When the camera disconnects during an inspection cycle, the error is not always correctly logged to `data/logs/`.  
**Fix:** Catch this in `camera.py` with explicit error logging and a clear ERROR status in the inspection result.

---

## Future development (roadmap)

### 1. Run stability test (NFR2)
The 8-hour stability test (FTC requirement) was not carried out before delivery. Recommended: connect the system to the production line and let it run for at least one full working day without manual intervention. Monitor log files for memory leaks or frame drops.

### 2. Migration to FastAPI
The Flask web framework (synchronous) can become a bottleneck under high polling frequencies. FastAPI (asynchronous, ASGI) can improve throughput and dashboard responsiveness, particularly at higher TCP trigger frequencies.  
A proof of concept was already considered during the implementation phase; the architecture is well suited for this migration.

### 3. Expand dataset for the classifier
The current classifier was trained on a relatively small dataset of OK and defective products from a single product type. When new product types or packaging variants are introduced, the dataset must be expanded and the model retrained.  
`tools/capture_dataset.py` and `tools/train_classifier.py` are already in place for this purpose.

### 4. Database storage for inspection results
Inspection results are currently stored as JSONL files per day in `data/results/`. For long-term analysis and integration with a MES or ERP system, storage in a database (e.g. SQLite or PostgreSQL) is more suitable.  
`backend/output/result_writer.py` is the integration point for this extension.

### 5. Automatic model selection based on product type
At the moment the detection backend is set manually in the config. If MDE Automation starts inspecting multiple product types, automatic model selection based on a product number (from the TCP trigger signal or a barcode) is a logical next step.

### 6. Export function in the dashboard
The analytics dashboard shows results per day. An export button (CSV or PDF) would make it easier to generate reports for quality control or customer communication.

### 7. Enable GPU acceleration (CUDA)
The system currently runs on CPU. On an IPC with an NVIDIA GPU, CUDA can be enabled for the classifier and YOLO, which can reduce cycle time by 50–70%. This requires installation of the CUDA toolkit and the appropriate PyTorch version with GPU support.

---

*Last updated: 5 June 2026 – Pieter van Haaften*
