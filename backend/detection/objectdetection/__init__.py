from .classifier import ImageClassifier
from .preprocessing import preprocess, preprocess_to_pil
from .template_detector import TemplateDetector

# YOLO is lazily loaded through yolo_detector module
# We import the functions but not the module at load time
def extract_rois(frame):
    """Lazy wrapper that imports yolo_detector on first use."""
    from .yolo_detector import extract_rois as _extract_rois
    return _extract_rois(frame)

def run_object_detection(frame):
    """Lazy wrapper that imports yolo_detector on first use."""
    from .yolo_detector import run_object_detection as _run_object_detection
    return _run_object_detection(frame)

__all__ = [
    "ImageClassifier",
    "TemplateDetector",
    "extract_rois",
    "preprocess",
    "preprocess_to_pil",
    "run_object_detection",
    ]