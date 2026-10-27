"""
Train the dot-font classifier (font reader) on real camera images.

Labels come from the file names (camera software format
"UId_0_Time_<d>-<m>-<yyyy>_<h>-<min>-<s>_<ms>.jpg"):
  line 1 = year digit + day of year + capture time rounded to the minute,
           e.g. 1 Oct 2026 10:28:51 -> "6274 10:29"
  line 2 = "<BB|EXP> <date>"; BB or EXP is decided per image by which layout
           the current model fits best (--date, default 05/28).
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

from backend.detection.ocr.font_reader import FontReader, cells_from, ink_maps, segment, DEFAULT_MODEL

NAME = re.compile(r"Time_(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)_")
LINE1 = "#### ##:##"
LAYOUTS = {"BB": "BB ##/##", "EXP": "EXP ##/##"}
NEW_CLASSES = ["E", "X", "P"]
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
}


def synth_cells(n_codes, rng):
    """Random two-line codes in the dot font with rib shadows, through the same
    ink map and cell cutting as the real images -> (cells, labels)."""
    X, Y = [], []
    for _ in range(n_codes):
        d = lambda: str(rng.integers(10))
        prefix = "EXP" if rng.random() < 0.5 else "BB"
        lines = ["".join(d() for _ in range(4)) + " " + d() + d() + ":" + d() + d(), prefix + " " + d() + d() + "/" + d() + d()]
        ch = rng.uniform(15, 24); py = ch / 6.5; px = py * rng.uniform(0.9, 1.15); adv = 6 * px * rng.uniform(0.95, 1.1)
        slant = rng.uniform(-0.05, 0.25); r = py * rng.uniform(0.33, 0.55); gap = ch * rng.uniform(0.2, 0.6)
        W, H = int(adv * 11 + 60), int(2 * ch + gap + 50)
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
            X.append(cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)); Y.append(c)
    return X, Y


def label_line1(path):
    m = NAME.search(Path(path).name)
    if not m:
        return None, None
    d, mo, y, h, mi, s = map(int, m.groups())
    t = datetime.datetime(y, mo, d, h, mi, s) + datetime.timedelta(seconds=30)
    doy = datetime.date(y, mo, d).timetuple().tm_yday
    return f"{str(y)[-1]}{doy:03d} {t:%H:%M}", (y, mo, d, h, mi, s)


def flat(template_lines, text_lines):
    return [c for tl, xl in zip(template_lines, text_lines) for t, c in zip(tl, xl) if t != " "]


def crop_of(engine, img, max_width):
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, round(img.shape[0] * max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return engine._prepare(img)[0]


def best_layout(reader, crop, line1, date):
    """Orientation + BB/EXP layout that the current model fits best; returns (template, labels, cells, score)."""
    boosted, raw = ink_maps(crop)
    best = None
    for prefix, l2 in LAYOUTS.items():
        tpl = [LINE1, l2]
        labels = flat(tpl, [line1, f"{prefix} {date}"])
        for flip in (False, True):
            seg = segment(raw[::-1, ::-1].copy() if flip else raw, tpl)
            cells = cells_from(boosted, seg, flip)
            if len(cells) != len(labels):
                continue
            p = reader._predict(cells)
            known = [(i, reader.idx[c]) for i, c in enumerate(labels) if c in reader.idx]
            score = float(np.mean([np.log(p[i, j] + 1e-9) for i, j in known]))
            if best is None or score > best[3]:
                best = (tpl, labels, cells, score)
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

    files = sorted((p for p in args.images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")),
                   key=lambda p: label_line1(p)[1] or (0,))
    per_day = defaultdict(list)
    for p in files:
        l1, ts = label_line1(p)
        if l1:
            per_day[ts[:3]].append((p, l1))
    split = {}
    for day, items in per_day.items():
        for i, (p, l1) in enumerate(items):
            split[p] = "train" if i % 2 == 0 else "test"

    t0 = time.time(); samples = []; skipped = Counter()
    for p, l1 in ((p, l1) for items in per_day.values() for p, l1 in items):
        img = cv2.imread(str(p))
        if img is None:
            skipped["unreadable file"] += 1; continue
        crop = crop_of(engine, img, args.max_width)
        fit = best_layout(old, crop, l1, args.date)
        if fit is None or fit[3] < args.min_fit:
            skipped["print does not fit"] += 1; continue
        tpl, labels, cells, score = fit
        samples.append({"path": p, "split": split[p], "template": tpl, "labels": labels, "cells": cells,
                        "text": f"{l1} {tpl[1].split()[0]} {args.date}",   # e.g. "6274 09:23 BB 05/28"
                        "crop": crop, "fit": score})
    classes = list(old.classes) + [c for c in NEW_CLASSES if c not in old.classes]
    idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(1)
    X, Y = [], []
    for s in (s for s in samples if s["split"] == "train"):
        for cell, lab in zip(s["cells"], s["labels"]):
            reps = 40 if lab in NEW_CLASSES else 12
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
        kind = "EXP" if s["template"][1].startswith("EXP") else "BB"
        for name, rd in (("new", new), ("old", old)):
            if name == "old" and kind == "EXP":
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
              "train_images": sum(s["split"] == "train" for s in samples),
              "test_images": sum(s["split"] == "test" for s in samples),
              "train_cells": len(Y), "classes": classes,
              "line2": dict(Counter("EXP" if s["template"][1].startswith("EXP") else "BB" for s in samples)),
              "test_results": {k: dict(v) for k, v in res.items()}, "wrong_examples": wrong,
              "seconds": round(time.time() - t0)}
    print(json.dumps(report, indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())