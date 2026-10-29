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
    _ink_fraction,
    _ink_sums,
    _largest_proximity_cluster,
    _normalize_rect_angle,
    alternative_rects,
    deskew_crop,
    extend_rect_along_line,
    locate_text_region,
    row_profile_rect,
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


class SolidLineRetryTests(unittest.TestCase):
    """With the rib eraser on, Hough "lines" along the synthetic text rows
    are erased and the locator found the line at only 11 of 33 tilts (the
    one-line prints of the 514 test images: no_text_detected). Retrying
    with only solid lines erased (long_line_min_solid_px) finds them."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_text_rows_erased_as_lines_are_found_by_the_retry(self):
        cfg = dict(DYN_CFG, reject_long_lines=True, long_line_min_solid_px=40)
        missed = [t for t in range(-80, 81, 5) if locate_text_region(_frame_with_tilt(t), cfg) is None]
        self.assertEqual(missed, [])

    def test_retry_off_keeps_the_old_behaviour(self):
        cfg = dict(DYN_CFG, reject_long_lines=True, long_line_min_solid_px=0)
        found = sum(locate_text_region(_frame_with_tilt(t), cfg) is not None for t in range(-80, 81, 5))
        self.assertLess(found, 33)


class RowProfileRectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_matches_the_line_angle_and_deskews_upright(self):
        for theta in range(-80, 81, 10):
            frame = _frame_with_tilt(theta)
            loc = locate_text_region(frame, DYN_CFG)
            self.assertIsNotNone(loc, f"not located at tilt {theta}")
            rect = row_profile_rect(loc["ink_points"], DYN_CFG["padding_px"])
            diff = (rect["angle"] - loc["rotated_rect"]["angle"] + 90) % 180 - 90
            self.assertLess(abs(diff), 2.0, f"row angle {rect['angle']} vs rect {loc['rotated_rect']['angle']}")
            crop = deskew_crop(frame, rect, extra_padding_px=20)
            self.assertGreater(crop.shape[1], crop.shape[0], f"crop not landscape at tilt {theta}")
            self.assertTrue(_is_upright(crop), f"crop upside-down at tilt {theta}")

    def test_ignores_a_stroke_that_tilts_the_tightest_box(self):
        # Two level rows of dots plus a diagonal stroke (a rib remnant):
        # minAreaRect tilts with the stroke, the row profile stays level.
        xs = np.arange(0, 120, 4.0)
        rows = [np.c_[xs, np.full_like(xs, y)] for y in (0, 1, 2, 40, 41, 42)]
        stroke = np.c_[120 + np.arange(0, 60.0), 42 + np.arange(0, 60.0)]
        pts = np.vstack(rows + [stroke]).astype(np.float32) + 100
        tight = cv2.minAreaRect(pts)[2]
        self.assertGreater(min(abs(tight) % 90, 90 - abs(tight) % 90), 5)
        self.assertAlmostEqual(row_profile_rect(pts, 0)["angle"], 0.0, delta=0.5)


class LargestClusterTests(unittest.TestCase):
    def test_min_members_skips_a_heavier_group_with_too_few_components(self):
        # Two big blobs (a rib crossing) outweigh five character-sized ones.
        boxes = [(0, 0, 40, 40), (45, 0, 85, 40)] + [(300 + 20 * i, 0, 315 + 20 * i, 20) for i in range(5)]
        areas = [900, 900] + [150] * 5
        self.assertEqual(_largest_proximity_cluster(boxes, areas, 30), [0, 1])
        self.assertEqual(_largest_proximity_cluster(boxes, areas, 30, min_members=4), [2, 3, 4, 5, 6])
        self.assertEqual(_largest_proximity_cluster(boxes, areas, 30, min_members=6), [])


def _dotted_words(words, y=240, x0=150, pitch=18, word_gap=30, tilt=0.0):
    """Frame with words of 3x3 px dot blocks ("characters" 12 px wide, 20 px
    tall), words word_gap px apart; returns (frame, [(x_start, x_end)])."""
    canvas = np.full((FRAME_H, FRAME_W), BACKGROUND, np.uint8)
    spans, x = [], x0
    for n in words:
        start = x
        for _ in range(n):
            for dx in range(0, 12, 4):
                for dy in range(0, 20, 4):
                    canvas[y + dy:y + dy + 3, x + dx:x + dx + 3] = INK
            x += pitch
        spans.append((start, x - pitch + 12))
        x += word_gap - (pitch - 12)
    if tilt:
        rot = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), tilt, 1.0)
        canvas = cv2.warpAffine(canvas, rot, (FRAME_W, FRAME_H), borderValue=BACKGROUND)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), spans


class ExtendRectAlongLineTests(unittest.TestCase):
    """The locator's crop held only "18:51" of "6286 18:51" (a rib between
    the words): extend_rect_along_line lengthens the rect over the ink that
    continues the line."""

    def _rect(self, span, y=240, angle=0.0):
        p = DYN_CFG["padding_px"]
        return {"center": [(span[0] + span[1]) / 2, y + 10], "size": [span[1] - span[0] + 2 * p, 20 + 2 * p],
                "angle": angle}

    def test_extends_over_the_other_word(self):
        frame, (w1, w2) = _dotted_words((4, 5))
        wide = extend_rect_along_line(frame, self._rect(w2), DYN_CFG)
        self.assertIsNotNone(wide)
        p = DYN_CFG["padding_px"]
        left = wide["center"][0] - wide["size"][0] / 2 + p
        right = wide["center"][0] + wide["size"][0] / 2 - p
        self.assertAlmostEqual(left, w1[0], delta=3)
        self.assertAlmostEqual(right, w2[1], delta=3)

    def test_none_without_ink_beside_the_rect(self):
        frame, (w,) = _dotted_words((5,))
        self.assertIsNone(extend_rect_along_line(frame, self._rect(w), DYN_CFG))

    def test_stops_at_a_gap_wider_than_a_word_space(self):
        frame, (w1, w2) = _dotted_words((4, 5), word_gap=150)
        self.assertIsNone(extend_rect_along_line(frame, self._rect(w2), DYN_CFG))

    def test_follows_a_tilted_line(self):
        # Rotated 20 degrees CCW about the frame centre; the rect around the
        # second word (in the rotated frame) extends along the same line.
        tilt = 20.0
        frame, (w1, w2) = _dotted_words((4, 5), tilt=tilt)
        rot = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), tilt, 1.0)
        cx, cy = rot @ np.array([(w2[0] + w2[1]) / 2, 250, 1.0])
        p = DYN_CFG["padding_px"]
        rect = {"center": [cx, cy], "size": [w2[1] - w2[0] + 2 * p, 20 + 2 * p], "angle": -tilt}
        wide = extend_rect_along_line(frame, rect, DYN_CFG)
        self.assertIsNotNone(wide)
        self.assertAlmostEqual(wide["size"][0] - 2 * p, w2[1] - w1[0], delta=4)


class InkFilterTests(unittest.TestCase):
    """Moulded lettering beside the print passes the blackhat threshold
    (relief shading) but is not ink: components whose box has too few
    pixels clearly darker than the plastic (ink_min_dark_fraction) are
    dropped before clustering."""

    CFG = dict(DYN_CFG, ink_dark_ratio=0.6, ink_min_dark_fraction=0.04, ink_filter=True)

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def _frame(self):
        # A printed line of dark dot blocks (40 on 190) at y=300, and above
        # it heavier "embossing": strokes only a little darker than the
        # plastic (150: blackhat 40, but 0.79 x the background).
        gray = np.full((FRAME_H, FRAME_W), BACKGROUND, np.uint8)
        for i in range(8):
            for dx in range(0, 12, 4):
                for dy in range(0, 20, 4):
                    gray[300 + dy:303 + dy, 250 + 18 * i + dx:253 + 18 * i + dx] = INK
        for i in range(8):
            y = 190 + 9 * (i % 2)                       # staggered: not a row the rib eraser takes
            gray[y:y + 35, 250 + 28 * i:260 + 28 * i] = 150
            gray[y:y + 10, 250 + 28 * i:272 + 28 * i] = 150
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_shading_lettering_is_not_located(self):
        loc = locate_text_region(self._frame(), self.CFG)
        self.assertIsNotNone(loc)
        self.assertGreater(loc["bbox"][1], 240, f"bbox {loc['bbox']} includes the embossing")

    def test_without_the_filter_the_lettering_wins(self):
        loc = locate_text_region(self._frame(), dict(self.CFG, ink_min_dark_fraction=0.0))
        self.assertLess(loc["bbox"][1], 240)

    def test_ink_fraction(self):
        gray = np.full((50, 50), 190, np.uint8)
        gray[10:20, 10:20] = 40                         # 100 dark px
        gray[30:40, 30:40] = 160                        # shading: not dark
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        sums = _ink_sums(gray, blackhat, self.CFG)
        self.assertAlmostEqual(_ink_fraction(sums, 10, 10, 20, 20), 0.25, places=3)
        self.assertEqual(_ink_fraction(sums, 30, 30, 10, 10), 0.0)


class AlternativeRectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_next_best_cluster_besides_the_chosen_one(self):
        # A heavy line and, far below it, a lighter one: the locator picks
        # the heavy one, the lighter one is the alternative.
        frame = _frame_with_tilt(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(5):
            gray[400:412, 200 + 20 * i:210 + 20 * i] = INK
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        loc = locate_text_region(frame, DYN_CFG)
        self.assertLess(loc["rotated_rect"]["center"][1], 300)
        alts = alternative_rects(loc, 2, DYN_CFG["padding_px"])
        self.assertEqual(len(alts), 1)
        self.assertAlmostEqual(alts[0]["center"][1], 405.5, delta=2)
        self.assertAlmostEqual(alts[0]["angle"], 0.0, delta=1)


if __name__ == "__main__":
    unittest.main()
