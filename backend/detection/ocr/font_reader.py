"""
backend/detection/ocr/font_reader.py

Reads the printed code with a small classifier trained on THIS printer's
dot-matrix font, instead of a general-purpose OCR model.

Why: on real frames PaddleOCR read the "6" lying under a cap rib as "5" in
~70% of cycles (and the "3" under crossing ribs as "2") - confidently, so
no amount of voting helped. A general model has never seen this font with a
rib through it. This reader was trained on thousands of synthetic codes in
the printer's 5x7 dot font, rendered onto real cap backgrounds with the
measured rib profile laid through the print, plus the real characters from
the debug captures.

Honest test (leave-one-bottle-out: trained without the bottle it is tested
on, so it had never seen that print): 73 of 83 real frames read correctly,
including 31/31 of the "4113 08:26" frames PaddleOCR mostly read as 08:25
and 20/20 of the crossing-ribs "3030 22:42" frames.

Pipeline, per deskewed crop from the text locator:
  1. ink map via dot_text (flat-field, ribs removed by shape)
  2. residual skew: angle with the sharpest row profile (+/-30 deg)
  3. one text band per template line (one or two, equal height) fitted to
     the row profile; for a one-line template, a crop that clearly holds
     two lines is not read (a two-line print must never pass as one line)
  4. per band, a fixed-pitch character grid fitted to the column profile,
     using the template layout (spaces between fields may or may not be
     printed, e.g. "BB 05/26" vs "BB08/27")
  5. each cell -> 24x16 -> classifier (digits, B, :, /)
  6. steps 2-5 for the crop as-is and rotated 180 degrees; the orientation
     whose literal positions (B, :, /) fit best wins
  7. decode: digit positions take the best digit; the result is only
     trusted if the literal positions really look like B, : and / (a
     garbled cut or wrong orientation fails this) and every digit is
     confident enough.

Several formats on one crop (OCRead reads every configured format): pass
a PreparedCrop to read() and the work that doesn't depend on the format is
done once - ink map, deskew angle per orientation and angle search, line
bands per line count, and each line's cell grid and cells (line 1
"#### ##:##" is the same in every format); per format only its own line 2
grid is fitted. The results are identical to reading each format on its own.

Supports templates of one or two lines made of '#' digit positions,
spaces and literals the model knows (B : / with the current model; a
literal O is scored with the 0 class, see LITERAL_CLASS; e.g.
["#### ##:##", "BB ##/##"] or ["#### ##:##"]). Other templates -> not
supported (read() returns None); OCRead skips such a format.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from backend.detection.ocr.dot_text import clean_dot_text

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "font_reader.npz"
CELL_W, CELL_H = 16, 24
# Literals scored with another class: in this dot font the letter O and the
# digit 0 are the same glyph, so a literal "O" (e.g. "PROD ##/##") is scored
# with the "0" class. Only literal positions - a digit position is never
# mapped to or from a letter.
LITERAL_CLASS = {"O": "0"}


def literal_class(ch):
    """The classifier class a literal template character is scored with."""
    return LITERAL_CLASS.get(ch, ch)
# One-line template: the crop holds a second line when the best two-band fit
# scores more than this times the best one-band fit. On one line the two-band
# fit only splits that line (ratio 1.00 on the fixture crops); on two-line
# prints it was 1.21-1.57.
SECOND_LINE_RATIO = 1.1
# With PreparedCrop.print_ink_ratio set (OCRead's retries), only when that
# second band holds printed ink: its ink darker than print_ink_ratio x the
# plastic around it (PreparedCrop.band_print_ink) is at least this fraction
# of the other band's. Moulded lettering beside a
# one-line print ("P2144/1", "KKT") made a second band (ratio 1.11-1.35) of
# relief shading: 0.00-0.04 on the test images. A printed line 2 is ink:
# >= 0.62 on the 444 correctly read two-line prints, and 0.49-0.57 where
# line 2 was printed with dots missing (those must stay NOK, not be read as
# a one-line code).
SECOND_LINE_MIN_PRINT_INK = 0.2
# Character pitch searched by the cell grid, as a multiple of the band height.
# The upper bound was 1.05: wider prints (e.g. an EXP line 2 with a 31 px
# pitch on a 25 px band, 1.25) could not be fitted and got a grid squeezed
# over part of the line. The fixture crops measure 0.45-0.72.
PITCH_RANGE = (0.45, 1.4)
# Line 2 is printed with line 1's pitch: its grid may differ from line 1's
# pitch by at most this factor (the fixtures: 0.90-1.08). Without it a
# half-pitch grid (two cells per glyph) won on dense letters like "EXP".
LINE_PITCH_TOLERANCE = 1.2


# -- geometry (identical to the evaluated prototype) --------------------------

def _rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _deskew(ink, max_deg=30.0, first_deg=8.0, cache=None):
    """Residual skew: the angle with the sharpest row profile. The locator
    already levelled the crop, so search +/-first_deg first and only widen
    to +/-max_deg when the best angle lies at the edge of that range.
    cache: dict angle -> sharpness for this ink, shared between calls (the
    searches overlap; all angles are multiples of 0.5, so keys are exact)."""
    cache = {} if cache is None else cache

    def sharp(a):
        a = float(a)
        if a not in cache:
            r = _rotate(ink, a).sum(axis=1)
            cache[a] = float((np.diff(r) ** 2).sum())
        return cache[a]
    coarse = max(np.arange(-first_deg, first_deg + 1e-6, 2.0), key=sharp)
    if abs(coarse) >= first_deg - 1e-6 and max_deg > first_deg:
        coarse = max(np.arange(-max_deg, max_deg + 1e-6, 2.0), key=sharp)
    return float(max(np.arange(coarse - 2.0, coarse + 2.01, 0.5), key=sharp))


def _fit_lines(fit, n_lines=2):
    """Best n_lines (1 or 2) equal-height text bands: (score, [(y0, y1), ...])."""
    r = fit.sum(axis=1); H = len(r); t = 0.25 * r.max(); c = np.r_[0.0, np.cumsum(r - t)]
    best = (-1e18, None)
    for h in range(8, min(45, H // 2) + 1):
        if n_lines == 1:
            ys = np.arange(0, H - h + 1)
            s = c[ys + h] - c[ys]
            i = int(np.argmax(s))
            if s[i] > best[0]:
                best = (float(s[i]), [(int(ys[i]), int(ys[i] + h))])
            continue
        # All gaps g at once: row g, column y (padded with -inf past the
        # last y for that gap). The first maximum in (g, y) order is the
        # one the per-gap loop with a strict ">" kept.
        gs = np.arange(0, h + 1)
        n = H - 2 * h - gs + 1
        if n.max() <= 0:
            continue
        ys = np.arange(0, n.max())
        g, y = gs[:, None], ys[None, :]
        s = (c[ys + h] - c[ys])[None, :] + (c[np.minimum(y + 2 * h + g, H)] - c[np.minimum(y + h + g, H)])
        s[y >= n[:, None]] = -np.inf
        gi, i = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[gi, i] > best[0]:
            yb = int(ys[i])
            best = (float(s[gi, i]), [(yb, yb + h), (yb + h + int(gi), yb + 2 * h + int(gi))])
    return best


def _layouts(template_line):
    parts = template_line.split(" ")
    out = []
    for mask in range(2 ** (len(parts) - 1)):
        cells, labels = [], []
        for i, part in enumerate(parts):
            if i and (mask >> (i - 1)) & 1:
                cells.append("S"); labels.append(" ")
            cells += ["G"] * len(part); labels += list(part)
        out.append((cells, labels))
    return out


class _GridTerms:
    """The template-independent part of _fit_cells for one band of one fit
    image (and line 1's pitch): the pitches and offsets searched, and per
    cell index k the core / edge means of cell k at every (pitch, offset),
    plus the left / right margin terms. Line 2 of BB, EXP and PROD is
    fitted on the same band: with the terms shared, each cell's means are
    computed once instead of once per format (_fit_cells was half of a
    first read). k_min: the offsets are those of the grid with k_min cells
    - a superset of every grid with more cells (same start and step, a
    later stop), so a template with at least k_min cells takes its own
    columns of them and gets the same numbers as on its own grid."""

    def __init__(self, fit, band, near_pitch, k_min):
        y0, y1 = band; h = max(1, y1 - y0)
        p_lo, p_hi = PITCH_RANGE[0] * h, PITCH_RANGE[1] * h
        if near_pitch:
            p_lo = max(p_lo, near_pitch / LINE_PITCH_TOLERANCE)
            p_hi = min(p_hi, near_pitch * LINE_PITCH_TOLERANCE)
            if p_lo >= p_hi:                            # no overlap: search the plain range
                p_lo, p_hi = PITCH_RANGE[0] * h, PITCH_RANGE[1] * h
        col = fit[y0:y1].sum(axis=0) / h
        self.W = len(col)
        self.cs = np.r_[0.0, np.cumsum(col)]
        self.ps = np.arange(p_lo, p_hi, 0.25)
        self.k_min = k_min
        self.core, self.edge, self.left, self.right = [], [], None, {}
        if not len(self.ps):
            return
        rows = [np.arange(-0.5 * p, self.W - k_min * p * 0.8, 0.5) for p in self.ps]
        self.x0 = np.zeros((len(self.ps), max(len(r) for r in rows)))
        for r, xs in enumerate(rows):
            self.x0[r, :len(xs)] = xs
        self.p = self.ps[:, None]

    def mean(self, a, b):
        W, cs = self.W, self.cs
        a = np.clip(np.round(a).astype(int), 0, W); b = np.clip(np.round(b).astype(int), 0, W)
        return np.where(b > a, (cs[b] - cs[a]) / np.maximum(1, b - a), 0.0)

    def cells(self, K):
        """core, edge (lists of per-cell arrays, at least K long), left and
        right (the K-cell grid's) margin terms."""
        x0, p = self.x0, self.p
        while len(self.core) < K:
            a = x0 + len(self.core) * p
            self.core.append(self.mean(a + 0.15 * p, a + 0.85 * p))
            self.edge.append(self.mean(a - 0.08 * p, a + 0.08 * p))
        if self.left is None:
            self.left = self.mean(x0 - p, x0)
        if K not in self.right:
            self.right[K] = self.mean(x0 + K * p, x0 + (K + 1) * p)
        return self.core, self.edge, self.left, self.right[K]


def _fit_cells(fit, band, template_line, near_pitch=None, shared=None):
    """Best fixed-pitch grid for one line: (score, (labels, [(x0, x1), ...])).
    near_pitch: line 1's pitch (px) - this line's pitch is then searched
    within LINE_PITCH_TOLERANCE of it instead of PITCH_RANGE alone.
    shared: a dict kept per (fit, band, near_pitch) by the caller, holding
    the _GridTerms that several templates fitted on that band share (same
    result as without)."""
    # All pitches at once: row r = pitch ps[r], column j = offset x0 (each
    # pitch has its own offsets, padded with NaN). The score per (pitch,
    # offset) is computed element by element exactly as one pitch at a time,
    # and the first maximum in (layout, pitch, offset) order wins - the same
    # grid as the per-pitch loop, ~3x faster.
    # The layouts of one line (space printed or not) share the per-cell
    # terms: cell k at x0 + k * p is the same whatever kind it is, and a
    # layout's offsets are a prefix of those of the layout with the fewest
    # cells (same start and step, earlier stop). So the offsets and each
    # cell's core/edge means are computed once on the widest grid and each
    # layout uses its own columns of them (_GridTerms).
    layouts = _layouts(template_line)
    k_min = min(len(cells) for cells, _ in layouts)
    terms = shared.get("terms") if shared is not None else None
    if terms is None or terms.k_min > k_min:
        terms = _GridTerms(fit, band, near_pitch, k_min)
        if shared is not None:
            shared["terms"] = terms
    ps, W = terms.ps, terms.W
    if not len(ps):
        return (-1e18, None)
    x0 = terms.x0
    best = (-1e18, None)
    for cells, labels in layouts:
        K = len(cells)
        lens = np.array([len(np.arange(-0.5 * q, W - K * q * 0.8, 0.5)) for q in ps])
        n = int(lens.max())
        if n == 0:
            continue
        valid = np.arange(n)[None, :] < lens[:, None]
        core, edge, left, right = terms.cells(K)
        s = np.zeros_like(x0[:, :n])
        for k, kind in enumerate(cells):
            s += core[k][:, :n] if kind == "G" else -2.0 * core[k][:, :n]
            s -= 0.5 * edge[k][:, :n]
        s -= 2.0 * (left[:, :n] + right[:, :n])
        s /= K
        s[~valid] = -np.inf
        r, i = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[r, i] > best[0]:
            pr, xr = ps[r], x0[r, i]
            best = (float(s[r, i]), (labels, [(float(xr + k * pr), float(xr + (k + 1) * pr)) for k in range(K)]))
    return best


def line_pitch(fit):
    """Pitch (px) of a _fit_cells() result, None when it found no grid."""
    cells = fit[1][1] if fit[1] else []
    return cells[1][0] - cells[0][0] if len(cells) > 1 else None


def ink_maps(crop):
    """(boosted ink for the classifier, un-boosted ink for fitting the grid), 0..1."""
    dbg = {}
    out = clean_dot_text(crop, {"upscale": 1, "illumination": "divide"}, debug=dbg)
    boosted = (255.0 - out[..., 0].astype(np.float32)) / 225.0
    raw = dbg["text_ink"]
    raw = np.clip(raw / max(1e-6, float(np.percentile(raw, 99.5))), 0, 1).astype(np.float32)
    return boosted, raw


def segment(fit_ink, template, first_deg=8.0):
    ang = _deskew(fit_ink, first_deg=first_deg)
    f = _rotate(fit_ink, ang)
    score, bands = _fit_lines(f, len(template))
    fits = []
    for band, tl in zip(bands, template):
        fits.append(_fit_cells(f, band, tl, line_pitch(fits[0]) if fits else None))
    # line_fit: the grid-fit score per line (how well the template's cell
    # layout matches the column profile), for comparing formats.
    seg = {"angle": ang, "bands": bands, "lines": [g for _, g in fits],
           "line_fit": [s for s, _ in fits], "second_line": False}
    if len(template) == 1:
        seg["second_line"] = _fit_lines(f, 2)[0] > SECOND_LINE_RATIO * score
    return seg


def _cut_cells(b, band, line):
    """The classifier cells of one line: b the rotated boosted ink, band
    (y0, y1), line (labels, [(x0, x1), ...]) from _fit_cells()."""
    (y0, y1), (labels, cells) = band, line
    out = []
    for lab, (a, c) in zip(labels, cells):
        if lab == " ":
            continue
        x0, x1 = int(round(a)), int(round(c))
        pad_y = max(1, (y1 - y0) // 8)
        cell = b[max(0, y0 - pad_y): y1 + pad_y, max(0, x0): max(x0 + 1, x1)]
        out.append(cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
                   if cell.size else np.zeros((CELL_H, CELL_W), np.float32))
    return out


def cells_from(boosted, seg, flip):
    b = boosted[::-1, ::-1] if flip else boosted
    b = _rotate(b, seg["angle"])
    return [cell for band, line in zip(seg["bands"], seg["lines"]) for cell in _cut_cells(b, band, line)]


class ReadTimeout(Exception):
    """A PreparedCrop's deadline passed before a step could start."""


class PreparedCrop:
    """One crop, read with several templates: segment() and cells_from()
    for any template, with the work shared between templates done once and
    cached - the ink maps; per orientation the deskew angle per angle search
    (+/-8, +/-30 deg; a search that lands on an angle already seen shares
    everything after it) and the sharpness per angle tried (the two searches
    overlap); per angle the line bands per number of lines; per
    band the cell grid per (template line, line 1's pitch) and its cells,
    and the per-cell terms of the grid search the templates share
    (_GridTerms).
    Same results as segment() / cells_from() on the crop's ink maps.

    timings_ms: time per step summed over all reads of this crop (ink,
    deskew, band_fit, cell_fit, cells).
    deadline: time.perf_counter() value; a step that would start after it
    raises ReadTimeout instead (None = no limit). Checked before each step,
    so a read stops at most one step late (the longest, a grid fit, takes
    ~50-150 ms)."""

    def __init__(self, crop, deadline=None, print_ink_ratio=None):
        self.crop = crop
        self.deadline = deadline
        # Set: a second band without printed ink (SECOND_LINE_MIN_PRINT_INK)
        # is not a second line for a one-line template, which is then read
        # on the printed ink alone (print_only). None: any second band
        # refuses a one-line template.
        self.print_ink_ratio = print_ink_ratio
        self._print_only = None
        self.timings_ms = {}
        self._ink = None
        self._fit_ink, self._angles, self._rotated, self._bands, self._grids, self._cells = {}, {}, {}, {}, {}, {}
        self._sharpness = {}                            # per orientation: deskew angle -> sharpness
        self._grid_terms = {}                           # per (flip, angle, band, line 1's pitch): _fit_cells' shared terms
        self._print_ink = {}

    def _timed(self, step, fn, *a, **k):
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise ReadTimeout(f"time budget used up before {step}")
        t = time.perf_counter()
        out = fn(*a, **k)
        self.timings_ms[step] = round(self.timings_ms.get(step, 0.0) + (time.perf_counter() - t) * 1000, 1)
        return out

    def ink(self):
        """(boosted, raw) from ink_maps(), computed on first use."""
        if self._ink is None:
            self._ink = self._timed("ink", ink_maps, self.crop)
        return self._ink

    def _rot(self, which, flip, ang):
        key = (which, flip, ang)
        if key not in self._rotated:
            img = self.ink()[0 if which == "boosted" else 1]
            self._rotated[key] = _rotate(img[::-1, ::-1] if flip else img, ang)
        return self._rotated[key]

    def _lines(self, flip, ang, n_lines):
        key = (flip, ang, n_lines)
        if key not in self._bands:
            self._bands[key] = self._timed("band_fit", _fit_lines, self._rot("raw", flip, ang), n_lines)
        return self._bands[key]

    def segment(self, template, flip, first_deg=8.0):
        """segment() of the crop's fit ink, flipped 180 degrees or not."""
        if flip not in self._fit_ink:
            raw = self.ink()[1]
            self._fit_ink[flip] = raw[::-1, ::-1].copy() if flip else raw
        if (flip, first_deg) not in self._angles:
            self._angles[(flip, first_deg)] = self._timed("deskew", _deskew, self._fit_ink[flip], first_deg=first_deg,
                                                          cache=self._sharpness.setdefault(flip, {}))
        ang = self._angles[(flip, first_deg)]
        f = self._rot("raw", flip, ang)
        score, bands = self._lines(flip, ang, len(template))
        fits = []
        for band, tl in zip(bands, template):
            near = line_pitch(fits[0]) if fits else None
            key = (flip, ang, band, tl, near)
            if key not in self._grids:
                # The templates fitted on this band share their per-cell terms.
                shared = self._grid_terms.setdefault((flip, ang, band, near), {})
                self._grids[key] = self._timed("cell_fit", _fit_cells, f, band, tl, near, shared)
            fits.append(self._grids[key])
        seg = {"angle": ang, "bands": bands, "lines": [g for _, g in fits],
               "line_fit": [s for s, _ in fits], "second_line": False}
        if len(template) == 1:
            seg["second_line"] = self._lines(flip, ang, 2)[0] > SECOND_LINE_RATIO * score
        return seg

    def _print_mask(self):
        """Pixels of printed ink: darker than print_ink_ratio x a
        closing-based background (the locator's ink measure,
        text_locator._ink_sums), dilated by 1 px - the threshold cuts the
        blurred edge ring off each dot. Embossing is relief shading and is
        not that dark; printed ink, even a line with dots missing, is."""
        if None not in self._print_ink:
            def mask():
                gray = cv2.cvtColor(self.crop, cv2.COLOR_BGR2GRAY) if self.crop.ndim == 3 else self.crop
                g = gray.astype(np.float32)
                bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
                m = (g < self.print_ink_ratio * bg).astype(np.uint8)
                return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(np.float32)
            self._print_ink[None] = self._timed("print_ink", mask)
        return self._print_ink[None]

    def band_print_ink(self, flip, ang):
        """Printed ink (_print_mask) of the fit ink in the weaker of the two
        bands of the two-line fit at deskew angle ang, as a fraction of the
        stronger band's."""
        if flip not in self._print_ink:
            m = self.ink()[1] * self._print_mask()
            self._print_ink[flip] = m[::-1, ::-1].copy() if flip else m
        _, bands = self._lines(flip, ang, 2)
        rot = _rotate(self._print_ink[flip], ang)
        mass = [float(rot[y0:y1].sum()) for y0, y1 in bands]
        return min(mass) / max(max(mass), 1e-9)

    def print_only(self):
        """The same crop with only its printed ink (_print_mask) in the ink
        maps: for a one-line template whose second band is not print."""
        if self._print_only is None:
            m = self._print_mask()
            boosted, raw = self.ink()
            self._print_only = PreparedCrop(self.crop, self.deadline)
            self._print_only._ink = (boosted * m, raw * m)
            self._print_only.timings_ms = self.timings_ms
        return self._print_only

    def cells(self, seg, flip):
        """cells_from() for a seg from segment(): each line's cells cut once."""
        out = []
        for band, line in zip(seg["bands"], seg["lines"]):
            key = (flip, seg["angle"], band, tuple(line[0]), tuple(line[1]))
            if key not in self._cells:
                self._cells[key] = self._timed("cells", _cut_cells, self._rot("boosted", flip, seg["angle"]),
                                               band, line)
            out += self._cells[key]
        return out


# -- classifier -----------------------------------------------------------------

class FontReader:
    def __init__(self, model_path=None, min_literal_score=-1.0, min_digit_confidence=0.0,
                 early_exit_literal=-0.25):
        data = np.load(str(model_path or DEFAULT_MODEL), allow_pickle=False)
        self.W1, self.b1, self.W2, self.b2 = (data[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
        self.classes = [str(c) for c in data["classes"]]
        self.idx = {c: i for i, c in enumerate(self.classes)}
        self.digits = [self.idx[str(d)] for d in range(10)]
        self.min_literal_score = float(min_literal_score)
        self.min_digit_confidence = float(min_digit_confidence)
        # Skip the upside-down pass when the upright read's B, : and / are
        # already clearly right (a wrong orientation scores far lower: -3 to
        # -11 on real frames). None = always try both orientations.
        self.early_exit_literal = early_exit_literal

    def supports(self, template) -> bool:
        return all(ch == "#" or ch == " " or literal_class(ch) in self.idx for line in template for ch in line)

    def _predict(self, cells):
        x = np.asarray(cells, np.float32).reshape(len(cells), -1)
        x = x / np.maximum(x.max(axis=1, keepdims=True), 1e-3)
        h = np.maximum(0, x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def _decode(self, probs, template):
        positions = [ch for line in template for ch in line if ch != " "]
        chars, conf, lit = [], 1.0, []
        for pos, p in zip(positions, probs):
            if pos == "#":
                j = self.digits[int(np.argmax(p[self.digits]))]
                chars.append(self.classes[j]); conf = min(conf, float(p[j]))
            else:
                chars.append(pos); lit.append(float(np.log(p[self.idx[literal_class(pos)]] + 1e-9)))
        it = iter(chars)
        lines = ["".join(" " if ch == " " else next(it) for ch in line) for line in template]
        return lines, (float(np.mean(lit)) if lit else 0.0), conf

    def read(self, crop, template):
        """Returns dict(lines, text, literal_score, confidence, flipped, angle, trusted) or None.

        crop: the image, or a PreparedCrop of it to share the work between
        several templates read on the same crop (same results).
        None also when a one-line template is read on a crop that holds two
        lines: that would be line 1 with line 2 ignored, not a reading. With
        the crop's print_ink_ratio set, a second band without printed ink
        (embossing beside the print) is not a second line: the one-line
        template is read on the printed ink alone."""
        prep = crop if isinstance(crop, PreparedCrop) else PreparedCrop(crop)
        if (prep.crop is None or prep.crop.size == 0 or not self.supports(template)
                or len(template) not in (1, 2)):
            return None
        t0 = time.perf_counter()
        timings = {}
        prep.ink()
        timings["ink_ms"] = round((time.perf_counter() - t0) * 1000, 1)   # 0 once the crop's ink is shared
        n_pos = sum(1 for line in template for ch in line if ch != " ")
        best = None
        for flip in (False, True):
            if (flip and best is not None and self.early_exit_literal is not None
                    and best["literal_score"] >= self.early_exit_literal):
                break                                   # upright read is clearly right
            ts = time.perf_counter()
            # Fast path: narrow angle search. If that read isn't clearly right,
            # redo it with the full +/-30 degree search (a wrong angle inside
            # the narrow range can otherwise win on a skewed crop).
            src = prep
            for first_deg in (8.0, 30.0):
                seg = src.segment(template, flip, first_deg=first_deg)
                if seg["second_line"]:
                    if (src is not prep or prep.print_ink_ratio is None
                            or prep.band_print_ink(flip, seg["angle"]) >= SECOND_LINE_MIN_PRINT_INK):
                        return None                     # two lines; orientation doesn't change that
                    # The second band is not print (embossing): from here on
                    # this orientation is read on the printed ink alone -
                    # unless that still holds two lines.
                    src = prep.print_only()
                    seg = src.segment(template, flip, first_deg=first_deg)
                    if seg["second_line"]:
                        return None
                cells = src.cells(seg, flip)
                if len(cells) != n_pos:
                    continue
                tp = time.perf_counter()
                lines, lit, conf = self._decode(self._predict(cells), template)
                timings["predict_ms"] = round(timings.get("predict_ms", 0.0) + (time.perf_counter() - tp) * 1000, 2)
                if best is None or lit > best["literal_score"]:
                    best = {"lines": lines, "literal_score": round(lit, 3), "confidence": round(conf, 3),
                            "flipped": flip, "angle": seg["angle"]}
                if self.early_exit_literal is None or lit >= self.early_exit_literal:
                    break
            timings["grid_down_ms" if flip else "grid_up_ms"] = round((time.perf_counter() - ts) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if best is None:
            return None
        best["timings_ms"] = timings
        best["text"] = " ".join(best["lines"])
        best["trusted"] = (best["literal_score"] >= self.min_literal_score
                           and best["confidence"] >= self.min_digit_confidence)
        return best
