"""
Tests for backend/detection/ocr/dot_text.py (used by the font reader).

Guards against the "4113 08:2" failure: a cap rib's dark shadow edge running
through a printed character, and rib handling that erases the character
along with the rib. Synthetic crops only - no OCR engine needed.

Run with:  python -m unittest tests.test_dot_text
"""
import unittest

import cv2
import numpy as np

from backend.detection.ocr.dot_text import clean_dot_text

BG = 150          # cap plastic
INK = 40          # printed ink
RIB_TOP = 200     # light top of a molded rib
RIB_EDGE = 95     # dark shadow line along the rib
H, W = 150, 260   # crop size at camera resolution
FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
BASELINE_Y = 70
PREFIX, LAST = "4113 08:2", "6"


def _text_layout():
    prefix_w = cv2.getTextSize(PREFIX, FONT, SCALE, THICK)[0][0]
    x0 = 40
    return x0, x0 + prefix_w + 2  # x of the prefix, x of the last character


def _crop(with_rib=True, with_text=True):
    img = np.full((H, W), BG, np.uint8)
    x0, x_last = _text_layout()
    if with_rib:
        # Vertical rib whose dark left edge runs through the last
        # character's left stroke - the geometry of the real "6".
        edge_x = x_last + 2
        cv2.rectangle(img, (edge_x + 2, 0), (edge_x + 9, H - 1), RIB_TOP, -1)
        cv2.rectangle(img, (edge_x, 0), (edge_x + 1, H - 1), RIB_EDGE, -1)
        # A long horizontal mold line well away from the text.
        cv2.line(img, (0, 125), (W - 1, 125), RIB_EDGE, 2)
    if with_text:
        cv2.putText(img, PREFIX, (x0, BASELINE_Y), FONT, SCALE, INK, THICK, cv2.LINE_AA)
        cv2.putText(img, LAST, (x_last, BASELINE_Y), FONT, SCALE, INK, THICK, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _last_char_box():
    _, x_last = _text_layout()
    (w, h), _ = cv2.getTextSize(LAST, FONT, SCALE, THICK)
    return x_last - 2, BASELINE_Y - h - 3, x_last + w + 3, BASELINE_Y + 4


def _dark_pixels(img_bgr, box, scale):
    x1, y1, x2, y2 = (v * scale for v in box)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return int((gray[y1:y2, x1:x2] < 128).sum())


class CleanDotTextTests(unittest.TestCase):
    def test_output_is_bgr_and_upscaled_like_the_existing_preprocessing(self):
        out = clean_dot_text(_crop())
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (H * 2, W * 2, 3))  # short side < 200 -> 2x

    def test_rib_and_mold_line_are_removed_away_from_the_text(self):
        out = cv2.cvtColor(clean_dot_text(_crop()), cv2.COLOR_BGR2GRAY)
        _, x_last = _text_layout()
        edge_x2 = (x_last + 2) * 2
        # Rib edge between the text and the crossing (camera rows 90-115),
        # and the mold line away from the crossing. Known limitation: right
        # where the dark mold line crosses the rib's bright top it has more
        # contrast than along the rest of its length, so a short dash can
        # remain at the crossing itself - excluded here.
        rib_below_text = out[180:230, edge_x2 - 2:edge_x2 + 6]
        mold_left = out[244:256, 20:edge_x2 - 30]
        mold_right = out[244:256, edge_x2 + 40:500]
        self.assertGreater(rib_below_text.min(), 200, "rib shadow edge survived")
        self.assertGreater(mold_left.min(), 200, "mold line survived")
        self.assertGreater(mold_right.min(), 200, "mold line survived")

    def test_character_printed_on_the_rib_edge_keeps_its_ink(self):
        box = _last_char_box()
        reference = _dark_pixels(clean_dot_text(_crop(with_rib=False)), box, 2)
        on_rib = _dark_pixels(clean_dot_text(_crop(with_rib=True)), box, 2)
        self.assertGreater(reference, 50)
        # Measured on this synthetic: the old derib + CLAHE path keeps ~48% of
        # the character's ink, dot_text ~71%. Ink on top of the shadow edge
        # isn't darker than the edge itself, so that part of the stroke comes
        # out mid-gray rather than black; the shape stays intact. 0.6 guards
        # against sliding back towards the derib behaviour. The upper bound
        # catches the rib edge being left inside the character instead.
        self.assertGreater(on_rib, 0.6 * reference, "character on the rib lost its ink")
        self.assertLess(on_rib, 1.3 * reference, "rib edge left inside the character")

    def test_rib_is_kept_when_line_removal_is_disabled(self):
        # Sanity check that the rib test above actually has a rib to remove.
        out = cv2.cvtColor(clean_dot_text(_crop(), {"line_min_length_px": 0}), cv2.COLOR_BGR2GRAY)
        _, x_last = _text_layout()
        edge_x2 = (x_last + 2) * 2
        self.assertLess(out[230:290, edge_x2 - 2:edge_x2 + 6].min(), 128)

    def test_blank_crop_is_white(self):
        out = clean_dot_text(_crop(with_rib=False, with_text=False))
        self.assertGreaterEqual(int(out.min()), 254)

    def test_config_values_override_defaults(self):
        debug = {}
        clean_dot_text(_crop(), {"line_angles": 4, "unknown_key": 1}, debug=debug)
        self.assertIn("lines", debug)




if __name__ == "__main__":
    unittest.main()
