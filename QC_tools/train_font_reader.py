"""
Train the dot-font classifier (font reader) on real camera images.

Labels come from the file names (camera software format
"UId_0_Time_<d>-<m>-<yyyy>_<h>-<min>-<s>_<ms>.jpg"):
  line 1 = year digit + day of year + capture time rounded to the minute,
           e.g. 1 Oct 2026 10:28:51 -> "6274 10:29"
  line 2 = "<BB|EXP> <date>"; BB or EXP is decided per image by which layout
           the current model fits best (--date, default 05/28).
Images whose print does not fit (cut off, unreadable) are left out.

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
    ap.add_argument("--min-fit", type=float, default=-1.5, help="leave out images the current model fits worse than this")
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
