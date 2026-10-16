from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)


def OCR(app_state=None):
    """
    Factory function that returns the configured OCR engine instance.
    Reads cfg["ocr"]["engine"] to select between "tesseract" (default) and "paddleocr".
    The returned object exposes a .run(frame) method with a consistent return shape.
    """
    engine = cfg.get("ocr", {}).get("engine", "tesseract").lower()

    if engine == "paddleocr":
        log.info("OCR engine: PaddleOCR")
        from backend.detection.ocr.paddle_ocr import PaddleOCR
        return PaddleOCR(app_state=app_state)

    log.info("OCR engine: Tesseract")
    from backend.detection.ocr.tesseract_ocr import TesseractOCR
    return TesseractOCR(app_state=app_state)


