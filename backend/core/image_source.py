"""
backend/core/image_source.py

Still images from a folder as a drop-in replacement for the camera
(camera.source: "images" in config/default.yaml). Same interface the rest of
the app uses - get_frame(), is_connected(), reconnect(), release() - so the
vision pipeline, the TCP/PLC trigger and the HMI video feed work unchanged.

step(+1 / -1) moves to the next / previous image (wrapping around). The
folder is re-read on every step, so images can be added while the app runs.
The HMI rotate button applies to the images too.

Images wider than max_width (default 744, the production camera's width) are
scaled down to that width, keeping the aspect ratio, because the OCRead
pipeline is tuned to the camera's character size. Smaller images are never
scaled up; max_width=0 keeps every image at its own size.

A file that can't be read is reported as "unreadable" in status(). While it
is the current file, get_frame() and frame_capture_time() return None, so a
read reports NOK (no frame) instead of reading the previous image under the
new file's name; the HMI video feed skips None frames and keeps showing the
previous image. is_connected() is True as long as the folder holds at least
one readable image, so a status poll in the middle of a step never sees the
source as disconnected.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from backend.utils import capture_time
from backend.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# AppState keeps the HMI rotation as quarter turns clockwise (0-3).
_ROTATE = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


class ImageFolderSource:
    def __init__(self, folder, app_state=None, max_width=744):
        self.folder = Path(folder)
        self.app_state = app_state
        self.max_width = int(max_width or 0)
        self._lock = threading.Lock()
        self._files: list[Path] = []
        self._index = 0
        self._image = None
        self._unreadable: str | None = None
        self._connected = False   # folder has a readable image; read without the lock
        with self._lock:
            self._scan()
            self._load()
        if self._files:
            log.info("Test images: %d image(s) in %s", len(self._files), self.folder)
        else:
            log.warning("Test images: no images found in %s", self.folder)

    def _scan(self):
        current = self._files[self._index] if self._files else None
        files = sorted((p for p in self.folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: p.name.lower()) if self.folder.is_dir() else []
        self._files = files
        if current in files:                      # stay on the same image after a re-scan
            self._index = files.index(current)
        elif self._index >= len(files):
            self._index = 0

    @staticmethod
    def _read(path: Path):
        """Decode an image file, None when it can't be read. np.fromfile +
        imdecode instead of cv2.imread so non-ASCII paths work on Windows."""
        try:
            return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            return None

    def _load(self):
        """Load the current file. self._image is only replaced once the new
        image is fully read and scaled; an unreadable file keeps the old one
        for the video feed, but get_frame() gives no frame for it."""
        if not self._files:
            self._image, self._unreadable, self._connected = None, None, False
            return
        path = self._files[self._index]
        img = self._read(path)
        if img is None:
            log.warning("Test images: could not read %s", path.name)
            self._unreadable = path.name
            # still connected if any other file in the folder is readable
            self._connected = any(self._read(p) is not None for p in self._files if p != path)
            return
        if self.max_width > 0 and img.shape[1] > self.max_width:
            h, w = img.shape[:2]
            img = cv2.resize(img, (self.max_width, round(h * self.max_width / w)),
                             interpolation=cv2.INTER_AREA)
        self._image, self._unreadable, self._connected = img, None, True

    def step(self, delta: int) -> dict:
        """Go to the next (+1) or previous (-1) image; returns status()."""
        with self._lock:
            self._scan()
            if self._files:
                self._index = (self._index + int(delta)) % len(self._files)
            self._load()
        return self.status()

    def status(self) -> dict:
        with self._lock:
            name = self._files[self._index].name if self._files else None
            return {"source": "images", "name": name, "index": self._index + 1 if self._files else 0,
                    "count": len(self._files), "folder": str(self.folder),
                    "unreadable": self._unreadable}

    # --- camera interface -------------------------------------------------
    def get_frame(self):
        with self._lock:
            img = None if self._image is None or self._unreadable else self._image.copy()
        if img is None:
            return None
        quarter_turns = int(self.app_state.get_camera_rotation()) % 4 if self.app_state is not None else 0
        return cv2.rotate(img, _ROTATE[quarter_turns]) if quarter_turns in _ROTATE else img

    def frame_capture_time(self):
        """Capture time of the current image from its file name
        (backend/utils/capture_time.py), None when the name has none or the
        current file is unreadable (get_frame() gives no frame then)."""
        with self._lock:
            if not self._files or self._unreadable:
                return None
            return capture_time.from_name(self._files[self._index])

    def is_connected(self) -> bool:
        return self._connected

    def reconnect(self):
        with self._lock:
            self._scan()
            self._load()

    def release(self):
        pass
