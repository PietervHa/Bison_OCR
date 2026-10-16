"""
Image Preprocessing Utilities

Functions for resizing and converting OpenCV frames for use with vision models.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a BGR OpenCV frame to (size, size), convert to RGB, and return uint8."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be > 0")

    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8, copy=False)


def preprocess_to_pil(frame: np.ndarray, size: int) -> Image.Image:
    """Apply preprocess() and return the result as an RGB PIL image."""
    rgb = preprocess(frame, size)
    return Image.fromarray(rgb, mode="RGB")


__all__ = ["preprocess", "preprocess_to_pil"]

