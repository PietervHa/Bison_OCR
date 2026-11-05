"""
Run the OCRead engine on every image in a folder, as the app does, and check
the reads against the labels from the file names.

The engine is the app's (backend/detection/ocread/engine.py): the configured
formats (ocr.read_validation.templates), the font reader gates and the value
check (value_pattern). Images are scaled down to camera.images_max_width as
the app's image source does (backend/core/image_source.py). Debug captures
(ocr.dynamic_roi.debug_save) are not written.

Expected codes come from the camera software's file name, with the same
rules as train_font_reader.py:
  line 1 = year digit + day of year + HH:MM; the printer's and camera's
           clocks are not in sync, so every line 1 candidate (capture time
           -60 s, +0 s, +30 s, see line1_candidates) counts as correct;
  line 2 = per format: BB / EXP with --date (the expiry date, default
           05/28), PROD with the capture month/year, or no line 2.
The print format isn't in the file name, so a read is correct when it
matches the expected code of any configured format.

Reports, per format the engine read (for a NOK: the format of its closest
rejected read, or "none") and in total: correct OK, WRONG OK, NOK with the
reasons, and the average and maximum time per image (the engine's
processing_time_ms: locate + read). The WRONG reads are listed with what was
read and what was expected. --save-nok DIR copies the NOK/WRONG images with overview sheets.\nThe JSON report (--report) also has every
image's read (text, outcome, reason, literal score, digit confidence,
formats tried), to compare two runs image by image.

--profile NAME reads with that product profile (config/profiles/NAME.yaml,
backend/core/profiles.py), as the app does after choosing it on the HMI;
without it, the config file as it is (= the standard profile).
--labels-from-folders takes the expected code from each image's folder
name instead of the file name, as train_font_reader.py does: e.g. folder
"6303 10-45 BB 05-28" (':' and '/' written as '-'); images in other folders
keep the file-name labels. Use it for shots of bottles printed earlier
(the file name's time is then not the print time).

Usage (from the repo root):
    python -m QC_tools.evaluate_images data/test_images
    python -m QC_tools.evaluate_images data/test_images --date 05/28 --report eval.json
    python -m QC_tools.evaluate_images data/test_images --first 100
    python -m QC_tools.evaluate_images data/blue_images --profile blue --labels-from-folders
"""
from __future__ import annotations

import argparse, json, logging, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from QC_tools.train_font_reader import capture_time, folder_label, format_name, label_lines, line1_candidates

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def expected_texts(t, date, formats):
    """Every code that counts as correct for capture time t: each format
    with each line 1 candidate, e.g. {"6288 10:27 PROD 10/26", ...}."""
    return {" ".join(label_lines(tpl, line1, date, t)) for tpl in formats for _, line1 in line1_candidates(t)}


def load_image(path, max_width):
    """The image as the app's image source gives it: scaled down to
    max_width (0 = keep size); None when unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None and max_width > 0 and img.shape[1] > max_width:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return img


def outcome(result, expected):
    """("correct" | "WRONG" | "NOK", read text, format name) for one engine result."""
    validation = result.get("validation") or {}
    fmt = validation.get("format")
    kind = format_name(fmt) if fmt else "none"
    text = " ".join(str(result.get("text") or "").split())
    if not validation.get("accepted"):
        return "NOK", text, kind
    return ("correct" if text in expected else "WRONG"), text, kind


def fmt_of(text):
    """Format name of an expected code: the first field of line 2, or "single_line"."""
    parts = text.split()
    return parts[2] if len(parts) > 2 else "single_line"


def summarize(rows):
    """rows: dicts with outcome, format, reason, ms. Returns {group: stats}
    for every format and "total"."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["format"]].append(r)
        groups["total"].append(r)
    out = {}
    for name in sorted(groups, key=lambda g: (g == "total", g)):
        rs = groups[name]
        n = Counter(r["outcome"] for r in rs)
        ms = [r["ms"] for r in rs]
        out[name] = {"images": len(rs), "correct_ok": n["correct"], "wrong_ok": n["WRONG"], "nok": n["NOK"],
                     "nok_reasons": dict(Counter(r["reason"] for r in rs if r["outcome"] == "NOK").most_common()),
                     "avg_ms": round(sum(ms) / len(ms), 1), "max_ms": round(max(ms), 1)}
    return out


def save_nok(rows, paths, outdir, per_sheet=24, cols=4, width=420):
    """Copy NOK/WRONG images to outdir/<reason>/ and write overview sheets:
    each tile = the image, its file name, outcome + reason, what was read
    (WRONG) and the expected code(s) from the file name."""
    import shutil
    import numpy as np
    bad = [r for r in rows if r["outcome"] in ("NOK", "WRONG")]
    tiles = []
    for r in bad:
        reason = "WRONG" if r["outcome"] == "WRONG" else (r["reason"] or "unknown")
        dest = outdir / str(reason)
        dest.mkdir(parents=True, exist_ok=True)
        src = paths[r["image"]]
        shutil.copy2(src, dest / src.name)
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((300, width, 3), 200, np.uint8)
        img = cv2.resize(img, (width, max(1, round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
        exp = [e for e in r["expected"] if fmt_of(e) == r["format"]] or r["expected"]
        lines = [src.stem.replace("UId_0_Time_", ""),
                 f"{r['outcome']}: {reason}" + (f"  read {r['text']}" if r["outcome"] == "WRONG" else ""),
                 "expected: " + " | ".join(exp[:2])]
        band = np.full((16 * len(lines) + 6, width, 3), 255, np.uint8)
        for k, t in enumerate(lines):
            cv2.putText(band, t[:58], (4, 15 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 200) if k == 1 else (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, img]))
    for sheet, start in enumerate(range(0, len(tiles), per_sheet), 1):
        chunk = tiles[start:start + per_sheet]
        h = max(t.shape[0] for t in chunk)
        chunk = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 6, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in chunk]
        while len(chunk) % cols:
            chunk.append(np.full_like(chunk[0], 255))
        grid = np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)])
        cv2.imwrite(str(outdir / f"nok_overview_{sheet}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("--date", default="05/28",
                    help="date on line 2 of BB and EXP, e.g. 05/28 (PROD: month/year of the capture date)")
    ap.add_argument("--report", type=Path, default=None, help="also write the report as JSON")
    ap.add_argument("--first", type=int, default=0, help="only the first N images (in file name order; 0 = all)")
    ap.add_argument("--save-nok", type=Path, default=None, metavar="DIR",
                    help="copy every NOK and WRONG image to DIR/<reason>/ and write overview sheets "
                         "(image + reason + expected code) to check whether a NOK was a bad print or a miss")
    ap.add_argument("--profile", default=None, metavar="NAME",
                    help="read with product profile NAME (config/profiles/NAME.yaml); default: the config file")
    ap.add_argument("--labels-from-folders", action="store_true",
                    help="expected code from the folder name, e.g. '6303 10-45 BB 05-28' (':' and '/' as '-')")
    args = ap.parse_args(argv)

    from backend.core.config_loader import cfg
    from backend.detection.ocread.engine import OCRead
    max_width = int((cfg.get("camera", {}) or {}).get("images_max_width", 744) or 0)
    if args.profile:
        from backend.core.profiles import config_for
        engine = OCRead(config_for(args.profile), profile_name=args.profile)
    else:
        engine = OCRead()
    engine.dynamic_roi_debug_dir = None                 # no debug captures for every image
    formats = [v.lines for v in engine.validators]      # ocr.read_validation.templates
    logging.disable(logging.WARNING)                    # not the engine's per-frame warnings

    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.first > 0:
        files = files[:args.first]
    rows, skipped = [], Counter()
    paths = {}
    for i, p in enumerate(files, 1):
        t = capture_time(p)
        folder = folder_label(p.parent.name, formats) if args.labels_from_folders else None
        if t is None and folder is None:
            skipped["no capture time in the name"] += 1; continue
        img = load_image(p, max_width)
        if img is None:
            skipped["unreadable file"] += 1; continue
        result = engine.read(img, capture_time=t)        # as test-image mode: the time in the file name
        expected = {" ".join(folder[1])} if folder else expected_texts(t, args.date, formats)
        kind, text, fmt = outcome(result, expected)
        read = ((result.get("validation") or {}).get("reads") or [{}])[0]
        paths[p.name] = p
        rows.append({"image": p.name, "outcome": kind, "format": fmt, "text": text,
                     "reason": result.get("failure_reason") or result.get("error"),
                     "expected": sorted(expected), "ms": float(result.get("processing_time_ms") or 0.0),
                     "literal_score": read.get("literal_score"), "confidence": read.get("min_token_confidence"),
                     "formats_tried": read.get("formats_tried")})
        if i % 25 == 0 or i == len(files):
            n = Counter(r["outcome"] for r in rows)
            print(f"{i}/{len(files)}  correct {n['correct']}  WRONG {n['WRONG']}  NOK {n['NOK']}  "
                  f"avg {sum(r['ms'] for r in rows) / len(rows):.0f} ms", file=sys.stderr, flush=True)
    if not rows:
        print(f"no labelled images in {args.images} (skipped: {dict(skipped)})")
        return 1

    summary = summarize(rows)
    wrong = [{"image": r["image"], "read": r["text"],
              # only the expected codes of the format that was read
              "expected": [e for e in r["expected"] if fmt_of(e) == r["format"]] or r["expected"]}
             for r in rows if r["outcome"] == "WRONG"]
    slowest = max(rows, key=lambda r: r["ms"])
    report = {"images": len(files), "evaluated": len(rows), "skipped": dict(skipped), "formats": formats,
              "profile": args.profile, "labels_from_folders": args.labels_from_folders,
              "max_width": max_width, "date": args.date, "results": summary,
              "slowest_image": slowest["image"], "wrong": wrong,
              "per_image": [{k: v for k, v in r.items() if k != "expected"} for r in rows]}

    print(f"profile: {args.profile or '(config file)'}")
    print(f"{'format':<12}{'images':>7}{'correct':>9}{'WRONG':>7}{'NOK':>6}{'avg ms':>9}{'max ms':>9}  NOK reasons")
    for name, s in summary.items():
        reasons = ", ".join(f"{k} {v}" for k, v in s["nok_reasons"].items())
        print(f"{name:<12}{s['images']:>7}{s['correct_ok']:>9}{s['wrong_ok']:>7}{s['nok']:>6}"
              f"{s['avg_ms']:>9.1f}{s['max_ms']:>9.1f}  {reasons}")
    if skipped:
        print(f"skipped: {dict(skipped)}")
    print(f"slowest: {slowest['image']} ({slowest['ms']:.1f} ms)")
    if wrong:
        print(f"\nWRONG ({len(wrong)}):")
        for w in wrong:
            print(f"  {w['image']}: read {w['read']!r}, expected {' | '.join(w['expected'])}")
    if args.save_nok:
        n_saved = save_nok(rows, paths, args.save_nok)
        print(f"\n{n_saved} NOK/WRONG image(s) saved to {args.save_nok} (see nok_overview_*.jpg)")
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
