"""backend/utils/roi.py

Shared ROI helpers used by both the OCR pipeline and the HMI video stream.

Having this logic in one place means any change to how fractional vs pixel
coordinates are resolved, or how edge-clamping works, only needs to be made
here — not in web.py AND paddle_ocr.py.
"""

from __future__ import annotations

import cv2
import numpy as np
from backend.utils.logger import get_logger

log = get_logger(__name__)


def roi_to_pixels(
    roi: dict,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """Convert a ROI config dict to clamped pixel coordinates.

    ROI values can be fractional (0.0–1.0, treated as a proportion of the
    frame dimension) or absolute pixel integers (> 1.0).

    Args:
        roi: dict with keys x_start, y_start, x_end, y_end.
        w:   frame width in pixels.
        h:   frame height in pixels.

    Returns:
        (x1, y1, x2, y2) clamped to the frame bounds, or
        (0, 0, 0, 0) when the resulting box is empty / invalid.
    """
    def _to_px(value: float, max_dim: int) -> int:
        if value <= 1.0:
            return int(round(value * max_dim))
        return int(round(value))

    x1 = _to_px(float(roi.get("x_start", 0.0)), w)
    y1 = _to_px(float(roi.get("y_start", 0.0)), h)
    x2 = _to_px(float(roi.get("x_end",   1.0)), w)
    y2 = _to_px(float(roi.get("y_end",   1.0)), h)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0

    return x1, y1, x2, y2


def draw_roi(frame: np.ndarray, roi: dict, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw the ROI rectangle onto *frame* in-place and return it.

    Args:
        frame:     BGR (or grayscale) frame to draw on.
        roi:       ROI config dict (x_start / y_start / x_end / y_end).
        color:     BGR colour tuple for the rectangle.
        thickness: Line thickness in pixels.

    Returns:
        The same frame with the rectangle drawn (no copy is made).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_pixels(roi, w, h)

    if x2 == 0 and y2 == 0:
        return frame  # empty box — nothing to draw

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def apply_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    """Crop *frame* to the region described by *roi*.

    Args:
        frame: BGR (or grayscale) input frame.
        roi:   ROI config dict (x_start / y_start / x_end / y_end).

    Returns:
        Cropped sub-frame, or the original frame when the ROI is invalid /
        results in an empty crop.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_pixels(roi, w, h)

    if x2 == 0 and y2 == 0:
        return frame  # empty box — return full frame unchanged

    cropped = frame[y1:y2, x1:x2]

    if cropped is None or cropped.size == 0:
        log.warning("ROI crop resulted in an empty frame; returning original.")
        return frame

    return cropped