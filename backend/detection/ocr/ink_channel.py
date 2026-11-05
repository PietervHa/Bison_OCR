"""
backend/detection/ocr/ink_channel.py

The image the print is found and read on, per product profile
(ocr.ink_channel in config/profiles/<name>.yaml).

Standard profile: no ink_channel - the camera frame goes in as it is and the
locator (text_locator.py) and the reader's ink map (dot_text.py) take its
grayscale, ink = darker than the plastic. Nothing here runs.

With ink_channel set, OCRead.read() turns the frame into one channel first,

    channel = w_B * B + w_G * G + w_R * R + offset    (saturated to 0..255)

optionally smoothed (blur_sigma, camera px), and hands that to the whole
pipeline as a grey BGR frame: locate, crop, ink map, retries, the agreement
check and the printed-ink mask all see the same channel. Ink must come out
DARK in it, like ink in a grey image.

Measured on the blue bottle (dynamic_roi debug captures 2026-11-05, ink
reddish-brown on blue-grey HD-PE): ink vs plastic next to it, and the
plastic's pixel noise -
    grey      -15.5  (noise 2.5)   the yellow rib glare and the HD-PE
                                   embossing shading are as dark/bright
    B - R     +4.6 warmer (7.2)    colour noise; the rib glare is warm too
    B         -13.3  (4.3)         crisp dots, the glare nearly gone, the
                                   embossing comes out light
So the blue profile uses B (weights_bgr [1, 0, 0]); warmth is
weights_bgr [1, 0, -1] with offset 128 (warm ink = dark).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class InkChannel:
    """One weighted sum of the B, G, R planes (+ offset), optionally blurred."""

    def __init__(self, weights_bgr, offset: float = 0.0, blur_sigma: float = 0.0):
        w = [float(v) for v in weights_bgr]
        if len(w) != 3:
            raise ValueError(f"ink_channel.weights_bgr needs 3 values (B, G, R), got {weights_bgr!r}")
        self.weights_bgr = w
        self.offset = float(offset)
        self.blur_sigma = float(blur_sigma)
        self._m = np.array([[w[0], w[1], w[2], self.offset]], np.float32)

    @classmethod
    def from_config(cls, c) -> Optional["InkChannel"]:
        """None (the grey frame, as the standard profile) for a missing or
        empty setting, else the channel. A typo must not silently fall back
        to grey: anything else that is not a valid setting raises."""
        if not c:
            return None
        if not isinstance(c, dict) or "weights_bgr" not in c:
            raise ValueError(f"ocr.ink_channel must be a mapping with weights_bgr, got {c!r}")
        unknown = set(c) - {"weights_bgr", "offset", "blur_sigma"}
        if unknown:
            raise ValueError(f"ocr.ink_channel: unknown keys {sorted(unknown)}")
        return cls(c["weights_bgr"], c.get("offset", 0.0), c.get("blur_sigma", 0.0))

    def describe(self) -> str:
        b, g, r = self.weights_bgr
        text = " ".join(f"{v:+g}{n}" for v, n in ((b, "B"), (g, "G"), (r, "R")) if v)
        if self.offset:
            text += f" {self.offset:+g}"
        return text + (f", blur {self.blur_sigma:g} px" if self.blur_sigma > 0 else "")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """The channel as a grey BGR uint8 image (all three planes equal, so
        every cv2 BGR2GRAY downstream gives the channel back exactly). A
        frame that is already single-channel is returned as it is."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame
        c = cv2.transform(frame, self._m)               # uint8, rounded and saturated
        if c.ndim == 3:
            c = c[..., 0]
        if self.blur_sigma > 0:
            c = cv2.GaussianBlur(c, (0, 0), self.blur_sigma)
        return cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
