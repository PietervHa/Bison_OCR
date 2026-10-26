"""
backend/core/image_source.py

Still images from a folder as a drop-in replacement for the camera
(camera.source: "images" in config/default.yaml). Same interface the rest of
the app uses - get_frame(), is_connected(), reconnect(), release() - so the
vision pipeline, the TCP/PLC trigger and the HMI video feed work unchanged.

step(+1 / -1) moves to the next / previous image (wrapping around). The
folder is re-read on every step, so images can be added while the app runs.
The HMI rotate button applies to the images too.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2

from backend.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# AppState keeps the HMI rotation as quarter turns clockwise (0-3).
_ROTATE = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


class ImageFolderSource:
    def __init__(self, folder, app_state=None):
        self.folder = Path(folder)
        self.app_state = app_state
        self._lock = threading.Lock()
        self._files: list[Path] = []
        self._index = 0
        self._image = None
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

    def _load(self):
        self._image = None
        if self._files:
            path = self._files[self._index]
            self._image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if self._image is None:
                log.warning("Test images: could not read %s", path)

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
                    "count": len(self._files), "folder": str(self.folder)}

    # --- camera interface -------------------------------------------------
    def get_frame(self):
        with self._lock:
            img = None if self._image is None else self._image.copy()
        if img is None:
            return None
        quarter_turns = int(self.app_state.get_camera_rotation()) % 4 if self.app_state is not None else 0
        return cv2.rotate(img, _ROTATE[quarter_turns]) if quarter_turns in _ROTATE else img

    def is_connected(self) -> bool:
        return self._image is not None

    def reconnect(self):
        with self._lock:
            self._scan()
            self._load()

    def release(self):
        pass
