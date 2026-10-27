"""
Train the dot-font classifier (font reader) on real camera images.

Labels come from the file names (camera software format
"UId_0_Time_<d>-<m>-<yyyy>_<h>-<min>-<s>_<ms>.jpg"):
  line 1 = year digit + day of year + time to the minute, e.g. "6274 10:29".
           The printer's and camera's clocks are not in sync, so the printed
           minute can differ by one from the capture time: the candidates are
           capture time -60 s, +0 s and +30 s (rounded down to the minute,
           duplicates removed), e.g. 1 Oct 2026 10:28:51 -> "6274 10:27",
           "6274 10:28", "6274 10:29". The candidate the current model fits
           best is used (chosen together with orientation and format), for
           training and as the truth in the test; the report counts how
           often each was chosen.
  line 2 = the format's second line with the date filled in (--date, default
           05/28), e.g. "BB 05/28" or "EXP 05/28"; empty for a single-line
           format. The formats come from the config
           (ocr.read_validation.templates); per image the format the current
           model fits best is used, a format with more lines first.
Images where the cutter fails clearly are left out; contact_sheet.png next to
the model shows training cells with their labels for a visual check.
Synthetic codes in the 5x7 dot font (incl. E, X, P) are added (--synth).

Half of the images per day are used for training, the other half is a test
set the new model never sees. The new model starts from the current one and
adds the letters E, X and P. Writes the model (.npz, same format as
backend/detection/ocr/models/font_reader.npz) and a JSON report.

Usage:
    python -m QC_tools.train_font_reader <images_dir> --out models_new/font_reader.npz
"""
from __future__ import annotations

import argparse, datetime, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from backend.detection.ocr.font_reader import FontReader, cells_from, ink_maps, literal_class, segment, DEFAULT_MODEL

NAME = re.compile(r"Time_(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)_")
NEW_CLASSES = ["E", "X", "P", "R", "D"]      # O is not a class: it is scored as 0
CELL_W, CELL_H = 16, 24

# 5x7 dot font for the synthetic training codes (as used for the first model).
FONT = {
    "0": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."], "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"], "3": [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."], "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."], "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."], "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."], ":": [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."],
    "/": [".....", "....#", "...#.", "..#..", ".#...", "#....", "....."], "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"], "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"], "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
}
FONT["O"] = FONT["0"]                               # same glyph in this dot font
SYNTH_PREFIXES = ["BB", "EXP", "PROD"]
SYNTH_SINGLE_LINE = 0.15                            # share of single-line synthetic codes


def synth_cells(n_codes, rng):
    """Random codes in the dot font with rib shadows (line 2 prefix BB / EXP /
    PROD, some single-line codes), through the same ink map and cell cutting
    as the real images -> (cells, labels). Labels are classifier classes
    (O -> 0)."""
    X, Y = [], []
    for _ in range(n_codes):
        d = lambda: str(rng.integers(10))
        prefix = SYNTH_PREFIXES[int(rng.integers(len(SYNTH_PREFIXES)))]
        lines = ["".join(d() for _ in range(4)) + " " + d() + d() + ":" + d() + d(), prefix + " " + d() + d() + "/" + d() + d()]
        if rng.random() < SYNTH_SINGLE_LINE:
            lines = lines[:1]
        ch = rng.uniform(15, 24); py = ch / 6.5; px = py * rng.uniform(0.9, 1.15); adv = 6 * px * rng.uniform(0.95, 1.1)
        slant = rng.uniform(-0.05, 0.25); r = py * rng.uniform(0.33, 0.55); gap = ch * rng.uniform(0.2, 0.6)
        W, H = int(adv * 11 + 60), int(len(lines) * ch + (len(lines) - 1) * gap + 50)
        base = rng.uniform(150, 215)
        img = np.full((H, W), base, np.float32) + np.linspace(0, rng.uniform(-25, 25), W)[None, :]
        S = 4; mask = np.zeros((H * S, W * S), np.float32); boxes = []
        x0, y0 = 25 + rng.uniform(0, 8), 22 + rng.uniform(0, 6)
        for li, text in enumerate(lines):
            y = y0 + li * (ch + gap); x = x0
            for c in text:
                if c != " ":
                    for rr, row in enumerate(FONT[c]):
                        for kk, on in enumerate(row):
                            if on == "#" and rng.random() > 0.04:
                                cy = y + rr * py + rng.normal(0, 0.15); cx = x + kk * px + slant * (6 - rr) * py + rng.normal(0, 0.15)
                                cv2.circle(mask, (int(cx * S), int(cy * S)), max(1, int(r * S * rng.uniform(0.85, 1.15))), 1.0, -1)
                    boxes.append((c, x - 0.5 * px, y - 0.5 * py, x - 0.5 * px + adv, y + 6.5 * py))
                x += adv
        m = cv2.GaussianBlur(cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA), (0, 0), rng.uniform(0.4, 1.1))
        img = img * (1 - rng.uniform(0.45, 0.7) * m)
        for _ in range(rng.choice([0, 1, 1, 2])):                 # rib: bright ridge with a shadow edge
            a = np.deg2rad(rng.uniform(0, 180)); cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.3, 0.7) * H
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            dist = (xx - cx) * -np.sin(a) + (yy - cy) * np.cos(a)
            img *= 1 + 0.12 * np.exp(-(dist / 2.0) ** 2) - 0.22 * np.exp(-((dist - 5) / 3.0) ** 2)
        img = np.clip(img + rng.normal(0, rng.uniform(1, 4), img.shape), 0, 255).astype(np.uint8)
        bgr = cv2.merge([np.clip(img.astype(np.int16) + 6, 0, 255).astype(np.uint8), img, img])   # slightly blue ink/plastic
        boosted = ink_maps(bgr)[0]
        for c, a0, b0, a1, b1 in boxes:
            w, h = a1 - a0, b1 - b0; jx, jy = rng.normal(0, 0.12 * w), rng.normal(0, 0.06 * h); pad = h / 8
            t, u = int(round(b0 + jy - pad)), int(round(b1 + jy + pad)); l, rgt = int(round(a0 + jx)), int(round(a1 + jx))
            cell = boosted[max(0, t):max(t + 1, u), max(0, l):max(l + 1, rgt)]
            X.append(cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)); Y.append(literal_class(c))
    return X, Y


LINE1_OFFSETS_S = (-60, 0, 30)


def capture_time(path):
    """Capture time from the camera software's file name, or None."""
    m = NAME.search(Path(path).name)
    if not m:
        return None
    d, mo, y, h, mi, s = map(int, m.groups())
    return datetime.datetime(y, mo, d, h, mi, s)


def line1_text(t):
    """Line 1 as printed at time t: year digit, day of year, HH:MM (seconds dropped)."""
    return f"{str(t.year)[-1]}{t.timetuple().tm_yday:03d} {t:%H:%M}"


def line1_candidates(t):
    """[(minutes vs. the capture minute, line 1 text)] for capture time t:
    t -60 s, t and t +30 s rounded down to the minute, duplicates removed.
    The day (and year digit) follow the shifted time, so 23:59:50 +30 s is
    the next day."""
    base = t.replace(second=0, microsecond=0)
    out = []
    for off in LINE1_OFFSETS_S:
        shifted = (t + datetime.timedelta(seconds=off)).replace(second=0, microsecond=0)
        cand = (int((shifted - base).total_seconds() // 60), line1_text(shifted))
        if cand not in out:
            out.append(cand)
    return out


def label_lines(template, line1, date):
    """Text per template line: line 1 from the file name, the '#' positions of
    any further line filled with the digits of --date ("BB ##/##" -> "BB 05/28")."""
    out = [line1]
    for tl in template[1:]:
        digits = iter(c for c in date if c.isdigit())
        try:
            out.append("".join(next(digits) if ch == "#" else ch for ch in tl))
        except StopIteration:
            raise ValueError(f"--date {date!r} has too few digits for {tl!r}") from None
    return out


def format_name(template):
    """Short name for reports: "BB", "EXP", ... (first field of line 2) or "single_line"."""
    return template[1].split()[0] if len(template) > 1 else "single_line"


def flat(template_lines, text_lines):
    return [c for tl, xl in zip(template_lines, text_lines) for t, c in zip(tl, xl) if t != " "]


def crop_of(engine, img, max_width):
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, round(img.shape[0] * max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return engine._prepare(img)[0]


def best_layout(reader, crop, line1_cands, date, formats, min_fit):
    """Orientation + format + line 1 candidate that the current model fits best;
    returns (template, labels, cells, score, line1, minute_delta) or None.
    line1_cands: [(minute_delta, text)] from line1_candidates().
    A format with more lines wins when its fit reaches min_fit (a two-line print
    also fits the single-line format on line 1; the font reader refuses that,
    and so does this). Like FontReader.read, the grid is fitted with the
    narrow (+/-8 deg) and the full (+/-30 deg) angle search; some crops only
    segment correctly with the full search."""
    boosted, raw = ink_maps(crop)
    best, best_key = None, None
    for tpl in formats:
        for flip, first_deg in ((f, d) for f in (False, True) for d in (8.0, 30.0)):
            seg = segment(raw[::-1, ::-1].copy() if flip else raw, tpl, first_deg=first_deg)
            if seg["second_line"]:
                continue
            cells = cells_from(boosted, seg, flip)
            p = None
            for delta, line1 in line1_cands:
                labels = flat(tpl, label_lines(tpl, line1, date))
                if len(cells) != len(labels):
                    continue
                if p is None:
                    p = reader._predict(cells)
                known = [(i, reader.idx[literal_class(c)]) for i, c in enumerate(labels) if literal_class(c) in reader.idx]
                score = float(np.mean([np.log(p[i, j] + 1e-9) for i, j in known]))
                key = (score >= min_fit, len(tpl), score)
                if best is None or key > best_key:
                    best, best_key = (tpl, labels, cells, score, line1, delta), key
    return best


def augment(cell, rng):
    h, w = cell.shape
    s = rng.uniform(0.9, 1.1); a = rng.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
    M[:, 2] += rng.uniform(-1.2, 1.2, 2)
    out = cv2.warpAffine(cell, M, (w, h), borderValue=0)
    out = out * rng.uniform(0.7, 1.2) + rng.normal(0, 0.04, out.shape)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.clip(out, 0, 1.5).astype(np.float32)


def train(X, y, n_cls, init, epochs=30, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    X = X.reshape(len(X), -1); X = X / np.maximum(X.max(1, keepdims=True), 1e-3)
    W1, b1 = init["W1"].copy(), init["b1"].copy()
    W2 = np.zeros((W1.shape[1], n_cls), np.float32); b2 = np.zeros(n_cls, np.float32)
    k = init["W2"].shape[1]
    W2[:, :k], b2[:k] = init["W2"], init["b2"]
    W2[:, k:] = rng.normal(0, 0.05, (W1.shape[1], n_cls - k))
    P = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; t = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), 256):
            idx = order[i:i + 256]; xb, yb = X[idx], y[idx]
            h = np.maximum(0, xb @ P[0] + P[1]); z = h @ P[2] + P[3]
            z -= z.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            g = pr; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            gW2 = h.T @ g; gb2 = g.sum(0); gh = (g @ P[2].T) * (h > 0)
            grads = [xb.T @ gh, gh.sum(0), gW2, gb2]
            t += 1
            for j in range(4):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]; v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                P[j] -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    return P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models_new/font_reader.npz"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--date", default="05/28", help="date on line 2, e.g. 05/28")
    ap.add_argument("--max-width", type=int, default=744)
    ap.add_argument("--min-fit", type=float, default=-4.0, help="leave out only clear failures (old model fit below this)")
    ap.add_argument("--synth", type=int, default=1500, help="synthetic codes to add (0 = none)")
    args = ap.parse_args(argv)

    from backend.detection.ocread.engine import OCRead
    engine = OCRead()
    for a in ("debug_dir", "dynamic_roi_debug_dir"):
        if hasattr(engine, a):
            setattr(engine, a, None)
    old = FontReader(DEFAULT_MODEL, min_literal_score=-1.0)
    formats = [v.lines for v in engine.validators]          # ocr.read_validation.templates

    files = sorted((p for p in args.images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")),
                   key=lambda p: capture_time(p) or datetime.datetime.min)
    per_day = defaultdict(list)
    for p in files:
        t = capture_time(p)
        if t:
            per_day[t.date()].append((p, t))
    split = {}
    for day, items in per_day.items():
        for i, (p, t) in enumerate(items):
            split[p] = "train" if i % 2 == 0 else "test"

    t0 = time.time(); samples = []; skipped = Counter(); no_fit = []; line1_chosen = Counter()
    for p, t in ((p, t) for items in per_day.values() for p, t in items):
        img = cv2.imread(str(p))
        if img is None:
            skipped["unreadable file"] += 1; continue
        crop = crop_of(engine, img, args.max_width)
        fit = best_layout(old, crop, line1_candidates(t), args.date, formats, args.min_fit)
        if fit is None or fit[3] < args.min_fit:
            skipped["print does not fit"] += 1; no_fit.append(p.name); continue
        tpl, labels, cells, score, l1, delta = fit
        line1_chosen[f"{delta:+d} min" if delta else "same minute"] += 1
        samples.append({"path": p, "split": split[p], "template": tpl, "labels": labels, "cells": cells,
                        "text": " ".join(label_lines(tpl, l1, args.date)),   # e.g. "6274 09:23 BB 05/28"
                        "crop": crop, "fit": score})
    classes = list(old.classes) + [c for c in NEW_CLASSES if c not in old.classes]
    idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(1)
    X, Y = [], []
    for s in (s for s in samples if s["split"] == "train"):
        for cell, lab in zip(s["cells"], s["labels"]):
            reps = 40 if lab in NEW_CLASSES else 12
            lab = literal_class(lab)                    # O -> 0 (same glyph)
            X.append(cell); Y.append(idx[lab])
            for _ in range(reps):
                X.append(augment(cell, rng)); Y.append(idx[lab])
    if args.synth:
        sx, sy = synth_cells(args.synth, rng)
        X += sx; Y += [idx[c] for c in sy]
    init = np.load(str(DEFAULT_MODEL))
    P = train(np.array(X, np.float32), np.array(Y), len(classes), init)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, W1=P[0], b1=P[1], W2=P[2], b2=P[3], classes=np.array(classes))

    new = FontReader(args.out, min_literal_score=-1.0)
    res = defaultdict(Counter); wrong = []
    for s in (s for s in samples if s["split"] == "test"):
        kind = format_name(s["template"])
        for name, rd in (("new", new), ("old", old)):
            if name == "old" and not old.supports(s["template"]):
                continue
            r = rd.read(s["crop"], s["template"])
            truth = " ".join(" ".join(x.split()) for x in [s["text"]])
            got = r and r["trusted"] and " ".join(r["text"].split())
            k = "correct" if got == truth else ("WRONG" if got else "NOK")
            res[f"{name}_{kind}"][k] += 1
            if name == "new" and k == "WRONG" and len(wrong) < 10:
                wrong.append((s["path"].name, truth, r["text"]))
    sheet = [s for s in samples if s["split"] == "train"][:: max(1, len([s for s in samples if s["split"] == "train"]) // 12)][:12]
    rows = []
    for smp in sheet:
        row = np.hstack([cv2.copyMakeBorder((255 - np.clip(c / max(c.max(), 1e-3), 0, 1) * 255).astype(np.uint8), 14, 2, 1, 1, cv2.BORDER_CONSTANT, value=255) for c in smp["cells"]])
        row = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
        for k, lab in enumerate(smp["labels"]):
            cv2.putText(row, lab, (k * (CELL_W + 2) + 4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        rows.append(row)
    if rows:
        wmax = max(r.shape[1] for r in rows)
        cv2.imwrite(str(args.out.parent / "contact_sheet.png"),
                    np.vstack([cv2.copyMakeBorder(r, 0, 3, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]))
    report = {"images": len(files), "used": len(samples), "skipped": dict(skipped),
              # Images that fit none of the configured formats (a new print
              # format shows up here instead of disappearing in "skipped").
              "no_fit_images": no_fit,
              # Which line 1 candidate was chosen, relative to the capture
              # minute (printer and camera clocks are not in sync).
              "line1_chosen": dict(line1_chosen),
              "train_images": sum(s["split"] == "train" for s in samples),
              "test_images": sum(s["split"] == "test" for s in samples),
              "train_cells": len(Y), "classes": classes,
              "formats": dict(Counter(format_name(s["template"]) for s in samples)),
              "test_results": {k: dict(v) for k, v in res.items()}, "wrong_examples": wrong,
              "seconds": round(time.time() - t0)}
    print(json.dumps(report, indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())