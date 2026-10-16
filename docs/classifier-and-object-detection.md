# Classifier training and object-detection model selection

This guide covers two separate workflows:

1. Training and using the **local image classifier**.
2. Switching the **object-detection backend** between different models/providers.

## Prerequisites

Install the project dependencies first:

```powershell
pip install -r requirements.txt
```

Run commands from the repository root:

```powershell
cd C:\Users\vanha\Documents\VisionSoftwareMDE
```

---

## 1) Training and using the local classifier

The local classifier is the `classifier` backend under `object_detection` in `config/default.yaml`.

### What the classifier expects

The training pipeline expects an image-folder dataset with this structure:

```text
data/dataset_split/
  train/
    ok/
    defective/
  val/
    ok/
    defective/
  test/
    ok/
    defective/
```

The training script uses the `train` and `val` splits. The `test` split is kept for your own later evaluation.

### Step 1: Capture images

Use the capture tool to collect labeled images:

```powershell
python .\tools\capture_dataset.py
```

During capture:

- Press `1` to save an `ok` sample.
- Press `2` to save a `defective` sample.
- Press `w` to rotate the preview by 90°.
- Press `q` to quit.

Images are saved under:

```text
data/dataset/ok/
data/dataset/defective/
```

### Step 2: Split the dataset

Create train/validation/test folders:

```powershell
python .\tools\split_dataset.py
```

By default, the splitter reads from `data/dataset` and writes to `data/dataset_split`.

### Step 3: Train the classifier

Train the model with the fine-tuning script:

```powershell
python .\tools\train_classifier.py
```

Useful options:

```powershell
python .\tools\train_classifier.py --epochs 10 --batch-size 8 --base-model facebook/convnextv2-tiny-22k-224
```

Training output is saved to:

```text
models/classifier/epoch-1/
models/classifier/epoch-2/
...
models/classifier/final/
```

The final folder contains the model and processor files used for inference.

### Step 4: Configure the app to use the classifier

Set the app to object-detection mode and point it to the classifier model path:

```yaml
vision_mode: "object_detection"

object_detection:
  backend: "classifier"
  classifier:
    model_path: "models/classifier/final"
```

Notes:

- Relative paths are resolved from the repository root.
- The classifier should include an `ok` label if you want the inspection engine to mark detections as `OK` only when the predicted label is `ok` and the confidence is above the threshold.
- The active confidence threshold is controlled by `confidence_threshold` in `config/default.yaml` or via the HMI/API.

### Step 5: Load the classifier at runtime

If the app is already running in maintenance mode, you can load a model without restarting:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:5000/load_classifier `
  -ContentType "application/json" `
  -Body '{"model_path":"models/classifier/final"}'
```

Check the active model with:

```powershell
Invoke-RestMethod http://localhost:5000/classifier_status
```

### Classifier runtime behavior

When the classifier backend is active:

- `confidence < threshold` always returns `NOK`.
- Predictions are evaluated against the `ok` label.
- If no model is loaded, the API returns `error: "no_model"`.

---

## 2) Switching object-detection models or providers

The object-detection stack supports three backends:

| Backend | What it uses | When to choose it |
| --- | --- | --- |
| `classifier` | Local image-classification model | You trained your own OK/defective model |
| `roboflow` | Roboflow hosted inference workflow | You want a remote provider / workflow-based detector |
| `yolo` | Local Ultralytics YOLO `.pt` model | You want a local general-purpose detector |

### Important setting

The backend is controlled here:

```yaml
object_detection:
  backend: "classifier"
```

The app validates this value at startup. If an unknown backend is provided, it falls back to `classifier`.

---

### A) Use a different local YOLO model

To use a different local YOLO checkpoint, switch the backend to `yolo` and set the model path:

```yaml
vision_mode: "object_detection"

object_detection:
  backend: "yolo"
  model_path: "models/yolov8n.pt"
  inference_size: 640
```

You can replace `models/yolov8n.pt` with any compatible YOLO checkpoint, for example:

```yaml
model_path: "models/my_custom_detector.pt"
```

Notes:

- `model_path` is used only by the YOLO backend.
- The model is loaded once on startup.
- YOLO uses the configured `inference_size` before converting boxes back to the original frame size.

---

### B) Use a different Roboflow model/profile

For Roboflow, the app uses a named profile stored under `roboflow.models`.

Example:

```yaml
vision_mode: "object_detection"

object_detection:
  backend: "roboflow"
  roboflow:
    model: "cola_detectie"
    models:
      cola_detectie:
        api_key: "YOUR_API_KEY"
        workspace: "your-workspace"
        workflow: "detect-count-and-visualize"
        api_url: "https://serverless.roboflow.com"
      another_profile:
        api_key: "YOUR_OTHER_API_KEY"
        workspace: "another-workspace"
        workflow: "another-workflow"
        api_url: "https://serverless.roboflow.com"
```

To switch provider/model, change only:

```yaml
roboflow:
  model: "another_profile"
```

Requirements for each Roboflow profile:

- `api_key`
- `workspace`
- `workflow`
- `api_url` is optional and defaults to `https://serverless.roboflow.com`

If `backend: roboflow` is selected and the profile is missing or incomplete, the app fails fast during config validation.

---

## 3) How the modes work together

There are two different choices in the app:

1. `vision_mode`
   - `ocr`
   - `object_detection`

2. `object_detection.backend`
   - `classifier`
   - `roboflow`
   - `yolo`

In other words:

- Set `vision_mode: "ocr"` if you want OCR.
- Set `vision_mode: "object_detection"` if you want inspection/object detection.
- Then choose which detection backend you want with `object_detection.backend`.

---

## 4) Recommended quick start

### Train your own classifier

1. Capture images.
2. Split the dataset.
3. Train the classifier.
4. Set `vision_mode: "object_detection"` and `object_detection.backend: "classifier"`.
5. Load the saved model from `models/classifier/final`.

### Switch to a different detector

- Use `backend: "yolo"` for a local `.pt` checkpoint.
- Use `backend: "roboflow"` and select a different profile in `roboflow.model`.

---

## 5) Useful API endpoints

- `GET /status` — current vision mode and machine status.
- `GET /classifier_status` — current classifier model path and loaded state.
- `POST /load_classifier` — load a classifier model at runtime.
- `POST /vision_mode` — change vision mode in maintenance mode.
- `POST /threshold` — change the active threshold in maintenance mode.

---

## 6) Troubleshooting

- **`model_path does not exist`**: check the path and remember that relative paths are resolved from the repo root.
- **`no_model`**: no classifier is loaded, or the selected backend does not support classifier loading.
- **Roboflow startup validation error**: verify that `roboflow.model` exists under `roboflow.models` and that the selected profile has non-empty `api_key`, `workspace`, and `workflow`.
- **Unexpected `NOK` results**: check `confidence_threshold` and confirm the model produces the expected label names.


