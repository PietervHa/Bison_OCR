"""backend/utils/roi.py

ROI coordinate helper used by the text locator (ocr.dynamic_roi.search_roi).
"""

from __future__ import annotations


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
