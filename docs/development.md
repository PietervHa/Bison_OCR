# Development Guide

This guide is for developers who want to contribute to or extend VisionSoftwareMDE.

## Setting Up Development Environment

### Prerequisites

- Python 3.10+
- Git
- Visual Studio Code or similar IDE
- Basic understanding of Python, Flask, PyTorch

### 1. Clone Repository

```powershell
git clone <repository-url>
cd VisionSoftwareMDE
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Development Dependencies

```powershell
pip install -r requirements.txt

# Additional dev tools
pip install pytest pytest-cov black pylint flake8 mypy
```

### 4. Verify Setup

```powershell
# Test imports
python -c "import backend.main; import frontend.web; print('OK')"

# Run a simple test
python -m pytest tests/camera_test.py -v
```

## Project Structure for Developers

```
VisionSoftwareMDE/
├── backend/
│   ├── main.py                 # Application entry point
│   ├── core/
│   │   ├── camera.py           # Camera interface
│   │   ├── config_loader.py    # Configuration system
│   │   ├── inspection_engine.py # Decision logic
│   │   ├── state.py            # Shared application state
│   │   ├── tcp_trigger_server.py # PLC interface
│   │   └── vision.py           # Vision pipeline orchestrator
│   ├── detection/
│   │   ├── ocr/               # OCR engines (Tesseract, PaddleOCR)
│   │   └── objectdetection/   # Object detection engines
│   ├── output/
│   │   └── result_writer.py   # Result persistence
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── roi.py             # ROI utilities
├── frontend/
│   ├── web.py                 # Flask web app
│   ├── static/                # CSS, JS, assets
│   └── templates/             # HTML templates
├── tests/
│   ├── camera_test.py
│   ├── paddleocr_test.py
│   ├── test_inspection_pipeline.py
│   └── images/                # Test fixtures
├── tools/
│   ├── capture_dataset.py     # Dataset capture tool
│   ├── split_dataset.py       # Dataset splitting
│   ├── train_classifier.py    # Classifier training
│   └── capture_template.py    # Template capture
└── docs/
    └── *.md                   # Documentation
```

## Code Style

### Formatting

Use Black for code formatting:

```powershell
# Format all backend code
black backend/

# Format specific file
black backend/core/vision.py
```

**Black Configuration** (`.flake8` or `pyproject.toml`):
```
line-length = 100
target-version = ['py310']
```

### Linting

Use Pylint to check code quality:

```powershell
# Check entire backend
pylint backend/

# Check specific file
pylint backend/core/camera.py --disable=missing-docstring
```

### Type Hints

Use type hints for better IDE support and maintainability:

```python
def process_frame(frame: np.ndarray, roi: Dict[str, float]) -> Dict[str, Any]:
    """Process frame and return detection results."""
    result: Dict[str, Any] = {}
    return result
```

### Docstrings

Write clear docstrings:

```python
def run_vision(frame: np.ndarray, callback: Optional[Callable] = None) -> None:
    """
    Trigger vision processing on a frame.
    
    Args:
        frame: Input image as numpy array (BGR format)
        callback: Optional function called with result dict when processing completes
        
    Returns:
        None - Results delivered via callback
        
    Raises:
        ValueError: If frame is invalid
    """
    pass
```

## Running Tests

### Run All Tests

```powershell
python -m pytest tests/ -v
```

### Run Specific Test File

```powershell
python -m pytest tests/camera_test.py -v
```

### Run with Coverage

```powershell
python -m pytest tests/ --cov=backend --cov-report=html
# Check htmlcov/index.html for coverage report
```

### Run Specific Test Function

```powershell
python -m pytest tests/camera_test.py::test_camera_initialization -v
```

## Writing Tests

Test template (`tests/test_example.py`):

```python
import pytest
from unittest.mock import Mock, patch
import numpy as np
from backend.core.inspection_engine import InspectionEngine

class TestInspectionEngine:
    """Test suite for InspectionEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create InspectionEngine instance for testing"""
        return InspectionEngine(app_state=None)
    
    def test_ok_detection(self, engine):
        """Test that detections above threshold mark as OK"""
        result = engine.determine_result(
            confidence=0.95,
            threshold=0.8,
            labels=["ok"]
        )
        assert result["status"] == "OK"
    
    def test_nok_detection_below_threshold(self, engine):
        """Test that low confidence marks as NOK"""
        result = engine.determine_result(
            confidence=0.5,
            threshold=0.8,
            labels=["ok"]
        )
        assert result["status"] == "NOK"
```

## Key Modules

### Camera (`backend/core/camera.py`)

**Interface:**
```python
from backend.core.camera import Camera

camera = Camera(index=0, app_state=app_state)
frame = camera.get_frame()  # Returns numpy array or None
```

**To extend:**
- Modify `Camera.__init__` for new camera types
- Override `get_frame()` for custom frame capture

### Vision Pipeline (`backend/core/vision.py`)

**Main entry point:**
```python
from backend.core.vision import run_vision, bind_app_state

bind_app_state(app_state)
run_vision(frame, callback=lambda result: handle_result(result))
```

**To extend:**
- Add new detection backend in `backend/detection/`
- Update vision mode dispatcher
- Register in `bind_app_state()`

### OCR Engines (`backend/detection/ocr/`)

**Interface:**
```python
ocr_engine = OCR()
result = ocr_engine.process_frame(frame, roi_coords)
# Returns: {status, confidence, keywords_found, dates_found, ...}
```

**To add new OCR engine:**
1. Create `backend/detection/ocr/your_engine.py`
2. Implement `process_frame(frame, roi_coords)` method
3. Update `ocr/__init__.py` dispatcher
4. Add config section in `default.yaml`

### Object Detection (`backend/detection/objectdetection/`)

**Interface:**
```python
detector = YOLODetector(model_path="models/yolov8n.pt")
result = detector.process_frame(frame, roi_coords)
# Returns: {objects: [class, confidence, box], ...}
```

**To add new backend:**
1. Create detector class
2. Implement required methods
3. Register in `backend/core/vision.py`
4. Add config validation

### Web API (`frontend/web.py`)

**Adding new endpoint:**
```python
def create_app(camera, app_state):
    app = Flask(__name__)
    
    @app.route('/api/custom', methods=['GET'])
    def custom_endpoint():
        """Custom API endpoint"""
        return {
            "status": "ok",
            "data": {"custom": "value"}
        }
    
    return app
```

## Debugging

### Enable Debug Mode

```python
# In backend/main.py
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    main()
```

### Use pdb (Python Debugger)

```python
from pdb import set_trace

def vision_cycle():
    set_trace()  # Execution pauses here
    # Step through with 'n' (next), 'c' (continue), etc.
```

### VSCode Debugging

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Backend Main",
            "type": "python",
            "request": "launch",
            "module": "backend.main",
            "console": "integratedTerminal"
        }
    ]
}
```

Then press F5 to start debugging.

### Logging

Use the centralized logger:

```python
from backend.utils.logger import get_logger

log = get_logger(__name__)
log.debug("Debug message")
log.info("Processing started")
log.warning("Frame is null")
log.error("Fatal error", exc_info=True)
```

**Log file:** `data/logs/app_YYYYMMDD.log`

## Common Development Tasks

### Add New Configuration Parameter

1. Add to `config/default.yaml`:
```yaml
new_setting:
  param1: value1
  param2: value2
```

2. Use in code:
```python
from backend.core.config_loader import cfg

param_value = cfg["new_setting"]["param1"]
```

3. Add validation in `config_loader.py`:
```python
def validate_config(cfg):
    # ... existing validations ...
    assert "new_setting" in cfg, "new_setting is required"
```

### Add New API Endpoint

1. Edit `frontend/web.py`:
```python
@app.route('/api/new_endpoint', methods=['POST'])
def new_endpoint():
    data = request.get_json()
    param = data.get('param')
    
    # Validation
    if not param:
        return {"status": "error", "error": "param required"}, 400
    
    # Processing
    result = process(param)
    
    return {"status": "ok", "data": result}
```

2. Document in `docs/API.md`

3. Test with:
```powershell
$body = @{param = "value"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:5000/api/new_endpoint `
  -Body $body -ContentType "application/json"
```

### Implement Custom Detection Backend

1. Create `backend/detection/objectdetection/my_detector.py`:
```python
class MyDetector:
    def __init__(self, **kwargs):
        # Initialize model
        pass
    
    def detect(self, frame, roi_coords):
        """Run inference on frame
        
        Returns: {
            'status': 'OK'/'NOK',
            'confidence': float,
            'objects': [...]
        }
        """
        pass
```

2. Register in `backend/core/vision.py`:
```python
if backend == "my_detector":
    detector = MyDetector(...)
    result = detector.detect(frame, roi)
```

3. Add validation and config

4. Add tests in `tests/`

## Benchmarking

### Run Performance Benchmarks

```powershell
python .\benchmarks\object_detection_benchmark.py
python .\benchmarks\ocr_benchmark.py
python .\benchmarks\hmi_polling_benchmark.py
```

Results saved to `benchmarks/benchmark_results/`

### Profile Code

```powershell
# Install profiler
pip install py-spy

# Profile running application
py-spy record -o profile.svg -- python -m backend.main

# Analyze with cProfile
python -m cProfile -s cumtime -m backend.main
```

### Memory Profiling

```powershell
pip install memory-profiler

# Profile specific function
python -m memory_profiler my_script.py
```

## Version Control Workflow

### Creating a Feature Branch

```powershell
git checkout -b feature/my-feature
git add .
git commit -m "Add new feature: description"
git push origin feature/my-feature

# Create pull request on GitHub/GitLab
```

### Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example:**
```
feat(ocr): add PaddleOCR support

- Implement PaddleOCR engine
- Add config options for PaddleOCR
- Update tests

Closes #123
```

### Before Committing

```powershell
# Format code
black backend/ frontend/

# Lint code
pylint backend/ frontend/ --disable=missing-docstring

# Run tests
python -m pytest tests/ -v

# Check types
mypy backend/
```

## Documentation

### Update Documentation

When adding features:

1. Add docstrings to functions/classes
2. Update relevant `.md` files in `docs/`
3. Add examples to API endpoint documentation
4. Update project README if scope changes

### Generate API Documentation

```powershell
# Using pdoc
pip install pdoc
pdoc --html backend/ -o docs/api

# Open docs/api/backend/index.html
```

## Troubleshooting Development

### Import Errors

```
ModuleNotFoundError: No module named 'backend'
```

**Solution:** Ensure you're running from project root with activated `.venv`:
```powershell
cd C:\Users\vanha\Documents\VisionSoftwareMDE
.\.venv\Scripts\Activate.ps1
```

### Camera Not Found in Tests

```powershell
# Skip camera tests
python -m pytest tests/camera_test.py -m "not requires_camera" -v
```

### Tests File Locking Issues

```powershell
# Close any open handles to data/ folder
# Restart PowerShell
# Retry test
```

## Performance Optimization Tips

1. **Profile first** - Use profilers before optimizing
2. **Reduce ROI size** - Smaller inspection area = faster
3. **Use faster models** - Nano vs. Large models
4. **Batch processing** - Process multiple frames together
5. **Cache models** - Load models once, reuse
6. **Async I/O** - Don't block on file writes
7. **GPU acceleration** - Use CUDA if available

## Common Pitfalls

### ❌ Don't:
- Modify global state without locks
- Ignore exceptions silently
- Use absolute paths (breaks on different machines)
- Block the main thread
- Load models repeatedly

### ✅ Do:
- Use `AppState` for thread-safe state
- Log all exceptions
- Use relative paths or config
- Run heavy operations in background threads
- Cache loaded models

## Useful References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Python Threading](https://docs.python.org/3/library/threading.html)

## Getting Help

- Check existing issues/discussions
- Review related code and tests
- Ask questions in team channels
- Create detailed issue report with:
  - Steps to reproduce
  - Expected vs. actual behavior
  - Environment info (OS, Python version)
  - Relevant logs from `data/logs/`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (not yet available) for contribution guidelines.

General workflow:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Format and lint code
5. Create pull request with description
6. Address review feedback
7. Merge when approved

