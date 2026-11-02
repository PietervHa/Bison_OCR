"""
backend/utils/capture_time.py

Capture time of a stored image from its file name, for test-image mode
(camera.source "images") and the QC tools. Recognised:
  - the camera software's  UId_<n>_Time_<d>-<m>-<yyyy>_<h>-<min>-<s>_<ms>.jpg
  - the app's own stamps   <yyyymmdd>_<hhmmss>[_<ms>] (debug capture folders)
None when the name has neither.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

_CAMERA = re.compile(r"Time_(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})-(\d{1,2})-(\d{1,2})(?:_|\.|$)")
_STAMP = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?!\d)")


def from_name(path) -> datetime.datetime | None:
    """The capture time in the file name of path, or None."""
    name = Path(str(path)).name
    try:
        m = _CAMERA.search(name)
        if m:
            d, mo, y, h, mi, s = map(int, m.groups())
            return datetime.datetime(y, mo, d, h, mi, s)
        m = _STAMP.search(name)
        if m:
            return datetime.datetime(*map(int, m.groups()))
    except ValueError:                              # e.g. day 31 of a 30-day month
        return None
    return None
