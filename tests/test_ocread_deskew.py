"""
Regression tests for OCRead's text localisation + deskew orientation.

Bug being guarded against: locate_text_region() used to report the
rotated-rect angle anywhere in (0, 180], so deskew_crop() "levelled" every
counter-clockwise-tilted line by turning it ~178 degrees - i.e. handed it to
OCR upside-down - while the same tilt clockwise came out upright. A synthetic
sweep from -85 to +85 degrees flipped 84 of 170 crops before the fix.

Run with:  python -m unittest tests.test_ocread_deskew
"""
import logging
import unittest

import cv2
import numpy as np

from backend.detection.ocr.text_locator import (
    _normalize_rect_angle,
    deskew_crop,
    locate_text_region,
)

# Fixed copy of the production ocr.dynamic_roi tuning at the time this test
# was written, so retuning config/default.yaml can't silently change what
# this test exercises. The test is about orientation, not about the tuning.
#
# One deliberate difference: reject_long_lines is off. With it on, the rib
# eraser (_erase_long_lines) treats the aligned tops/bottoms of a row of
# characters as a "rib" and erases them, and on this synthetic text the
# locator then often finds nothing at all - e.g. "BB0588311" is located at
# 0 of 33 tilts with it on vs 33 of 33 with it off. That is a separate open
# issue (see review item 7); keeping it out of this test means a failure
# here always points at the deskew orientation.
DYN_CFG = {
    "enabled": True,
    "search_roi": {"x_start": 0.0, "y_start": 0.0, "x_end": 1.0, "y_end": 1.0},
    "blackhat_kernel_size": 25,
    "blackhat_threshold": 32,
    "reject_long_lines": False,
    "long_line_min_length_px": 60,
    "long_line_erase_thickness_px": 6,
    "morph_kernel_w": 9,
    "morph_kernel_h": 5,
    "min_component_area": 3,
    "max_component_area": 9000,
    "max_component_dim": 70,
    "min_cluster_components": 4,
    "cluster_max_gap_px": 30,
    "line_extend_max_perp_px": 18,
    "line_extend_max_gap_px": 90,
    "padding_px": 25,
}

FRAME_H, FRAME_W = 480, 744  # DFK 22BUC03 sensor size
BACKGROUND, INK = 190, 40
# Heavy glyphs on the left, light ones on the right: an upright crop has
# its ink centre of mass left of centre, an upside-down crop right of it.
TEXT = "8888111"
CHAR_PITCH_PX = 24


def _render_upright(canvas, x0, y0):
    x = x0
    for ch in TEXT:
        cv2.putText(canvas, ch, (x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 2, cv2.LINE_AA)
        x += CHAR_PITCH_PX


def _frame_with_tilt(theta_deg):
    """Light 'cap' with one dark line of text, rotated by theta_deg
    (cv2 convention: positive = counter-clockwise on screen)."""
    canvas = np.full((FRAME_H, FRAME_W), BACKGROUND, np.uint8)
    text_w = CHAR_PITCH_PX * len(TEXT)
    _render_upright(canvas, (FRAME_W - text_w) // 2, FRAME_H // 2 + 6)
    rot = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), theta_deg, 1.0)
    canvas = cv2.warpAffine(canvas, rot, (FRAME_W, FRAME_H), borderValue=BACKGROUND)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _is_upright(crop):
    """True if the heavy end of the line is on the left of the crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = np.clip(BACKGROUND - gray.astype(np.float32), 0, None)
    total = ink.sum()
    if total <= 0:
        return False
    cx = (ink.sum(axis=0) * np.arange(ink.shape[1])).sum() / total
    return cx < ink.shape[1] / 2


class NormalizeRectAngleTests(unittest.TestCase):
    def test_angle_always_in_minus90_to_90(self):
        for angle in np.arange(0.5, 90.01, 0.5):
            for rw, rh in ((200.0, 40.0), (40.0, 200.0)):
                w, h, a = _normalize_rect_angle(rw, rh, float(angle))
                self.assertGreaterEqual(w, h)
                self.assertGreater(a, -90.0)
                self.assertLessEqual(a, 90.0)

    def test_small_ccw_tilt_gives_small_negative_angle(self):
        # What OpenCV reports for a wide line tilted ~3 degrees CCW: a
        # "portrait" rect at ~87 degrees. Old code turned this into 177.
        w, h, a = _normalize_rect_angle(40.0, 200.0, 87.0)
        self.assertEqual((w, h), (200.0, 40.0))
        self.assertAlmostEqual(a, -3.0)

    def test_small_cw_tilt_unchanged(self):
        w, h, a = _normalize_rect_angle(200.0, 40.0, 3.0)
        self.assertEqual((w, h, a), (200.0, 40.0, 3.0))


class DeskewOrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_upright_text_is_never_flipped_by_deskew(self):
        located = 0
        flipped = []
        for theta in range(-80, 81, 5):
            frame = _frame_with_tilt(theta)
            loc = locate_text_region(frame, DYN_CFG)
            if not loc:
                continue
            located += 1
            crop = deskew_crop(frame, loc["rotated_rect"], extra_padding_px=20)
            self.assertIsNotNone(crop, f"deskew_crop returned None at tilt {theta}")
            self.assertGreater(crop.shape[1], crop.shape[0], f"crop not landscape at tilt {theta}")
            if not _is_upright(crop):
                flipped.append(theta)

        # Guard against the test silently passing because nothing was found.
        self.assertGreaterEqual(located, 25, "text locator found too few of the synthetic lines")
        self.assertEqual(flipped, [], f"deskew turned upright text upside-down at tilts {flipped}")

    def test_no_flip_either_side_of_snap_tolerance(self):
        # deskew_crop skips rotation within 1.5 degrees; the old code
        # flipped the text as soon as a CCW tilt crossed that threshold.
        for theta in (1, 2, 3, -1, -2, -3):
            frame = _frame_with_tilt(theta)
            loc = locate_text_region(frame, DYN_CFG)
            if not loc:
                continue
            crop = deskew_crop(frame, loc["rotated_rect"], extra_padding_px=20)
            self.assertTrue(_is_upright(crop), f"crop upside-down at tilt {theta}")


if __name__ == "__main__":
    unittest.main()
