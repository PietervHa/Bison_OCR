"""
Compare OCRead pipelines on saved camera frames, using the real PaddleOCR
engine and the real OCRead code path (locate -> deskew -> preprocess ->
recognise -> orientation retry -> output filter -> read validation).

Pipelines:
  previous             derib + CLAHE, no validation (configuration before 2026-10)
  all_variants         validation + position vote over EVERY preprocessing variant,
                       including the OCR-pipeline-style reads (region, full_frame,
                       ocr_pipeline) - used to report each variant's own accuracy
  dot_text_raw         dot_text, no validation (any text = OK)
  validated_single     read validation, one read (first consensus variant)
  validated_consensus  read validation + consensus, as configured in default.yaml
  bridge=X             validated_consensus with dot_text bridge_sigma_px=X (--bridge)

What to look at, with --expected: "WRONG OK" is the number that matters in
production - a code reported as OK that is not what is printed. A reject is
safe (NOK, with its reason); a wrong OK is not. Without --expected you get
OK/NOK counts, reject reasons and the most common reads.

Getting frames: set ocr.dynamic_roi.debug_save: true, run a session, then
point this at data/debug/dynamic_roi - every cycle folder contains
text_locator_search.png, the raw frame OCR saw. Any folder of .png/.jpg
frames works too (--glob). Run once per bottle/expected code.

Usage (from the repo root):
    python -m QC_tools.compare_ocr_preprocessing data/debug/dynamic_roi --expected "4113 08:26 BB 05/26"
    python -m QC_tools.compare_ocr_preprocessing frames/ --glob "*.png" --bridge 0.5,1.0 --csv out.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def _norm(text) -> str:
    return "".join(str(text or "").split())


def _find_frames(root: Path, pattern: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    if pattern:
        return sorted(root.glob(pattern))
    frames = sorted(root.glob("**/text_locator_search.png"))
    if frames:
        return frames
    return sorted([*root.glob("**/*.png"), *root.glob("**/*.jpg"), *root.glob("**/*.jpeg")])


def _resolve_frames_root(given):
    """The folder to read, with a clear message instead of "no frames" when
    it doesn't exist. Default: the configured debug_dir (project-root
    relative). Also checks backend/<path>, where captures ended up when the
    app was started from backend/ before debug_dir was anchored to the root."""
    from backend.core.config_loader import cfg
    from backend.detection.ocr.paddle_ocr import PROJECT_ROOT, resolve_project_path

    configured = (cfg.get("ocr", {}).get("dynamic_roi", {}) or {}).get("debug_dir") or "data/debug/dynamic_roi"
    rel = Path(given) if given is not None else Path(configured)
    candidates = [rel] if rel.is_absolute() else [Path.cwd() / rel, Path(resolve_project_path(rel)),
                                                  PROJECT_ROOT / "backend" / rel]
    for c in candidates:
        if c.exists():
            return c
    print(f"Folder not found: {rel}")
    print("Looked in:\n  " + "\n  ".join(str(c) for c in candidates))
    return None


def _in_window(path, since, until):
    """Debug cycle folders are named YYYYMMDD_HHMMSS_mmm; filter on that time."""
    import re
    for part in Path(path).parts[::-1]:
        m = re.match(r"\d{8}_(\d{2})(\d{2})(\d{2})", part)
        if m:
            t = ":".join(m.groups())
            norm = lambda s: (s + ":00")[:8] if s and len(s) == 5 else s
            return (not since or t >= norm(since)) and (not until or t <= norm(until) if len(until or "") > 5
                                                         else not until or t[:5] <= until)
    return False


def _pipelines(engine, bridge_values):
    dot_cfg = dict(engine.dot_text_cfg or {})
    validator = engine.validator
    variants = list(engine.consensus_variants)
    need = engine.consensus_min_agree
    # Every pipeline sets ALL of these, so no setting leaks from one pipeline
    # into the next (they share one engine).
    fusion = dict(consensus_fusion=engine.consensus_fusion, consensus_early_stop=engine.consensus_early_stop)
    single = dict(consensus_fusion="exact", consensus_early_stop=True)
    p = [
        ("previous", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                          dot_text_enabled=False, derib_enabled=True, dot_text_cfg=dot_cfg, **single)),
        ("dot_text_raw", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                              dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg, **single)),
    ]
    if validator is not None:
        p += [
            ("validated_single", dict(validator=validator, consensus_variants=variants[:1], consensus_min_agree=1,
                                      dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg, **single)),
            ("validated_consensus", dict(validator=validator, consensus_variants=variants, consensus_min_agree=need,
                                         dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg, **fusion)),
        ]
        all_variants = sorted(engine.PREPROCESS_VARIANTS - {"dot", "classic_derib"})
        p.append(("all_variants", dict(validator=validator, consensus_variants=all_variants,
                                       consensus_min_agree=need, consensus_fusion="position_vote",
                                       consensus_early_stop=False,
                                       dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)))
        for b in bridge_values:
            p.append((f"bridge={b:g}", dict(validator=validator, consensus_variants=variants, consensus_min_agree=need,
                                            dot_text_enabled=True, derib_enabled=False,
                                            dot_text_cfg=dict(dot_cfg, bridge_sigma_px=b), **fusion)))
    else:
        print("Note: ocr.read_validation is disabled in the config - validated pipelines skipped.")
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path, nargs="?", default=None,
                        help="folder (searched recursively) or a single image; "
                             "default: the configured ocr.dynamic_roi.debug_dir")
    parser.add_argument("--expected", default=None, help='expected code, e.g. "4113 08:26 BB 05/26"')
    parser.add_argument("--glob", default=None, help="glob pattern inside the folder (default: debug_save frames)")
    parser.add_argument("--bridge", default="", help="comma-separated bridge_sigma_px values to add")
    parser.add_argument("--csv", type=Path, default=None, help="write one row per frame and pipeline")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N frames")
    parser.add_argument("--since", help="only cycles captured from HH:MM[:SS] (from the debug folder name)")
    parser.add_argument("--until", help="only cycles captured up to HH:MM[:SS]")
    parser.add_argument("--config", default=None, help="config YAML (read by backend.core.config_loader)")
    args = parser.parse_args(argv)

    root = _resolve_frames_root(args.frames)
    if root is None:
        return 1
    frames = _find_frames(root, args.glob)
    if args.since or args.until:
        frames = [f for f in frames if _in_window(f, args.since, args.until)]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"No frames found under {root}")
        print("Expected per-cycle folders containing text_locator_search.png (written while "
              "ocr.dynamic_roi.debug_save is true), or use --glob for other image names.")
        return 1
    print(f"Reading frames from {root}")

    from backend.detection.ocr.paddle_ocr import PaddleOCR

    engine = PaddleOCR(app_state=None)   # starts the real PaddleOCR worker
    engine.dynamic_roi_debug_dir = None  # don't write debug images while comparing
    pipelines = _pipelines(engine, [float(v) for v in args.bridge.split(",") if v.strip()])
    expected = _norm(args.expected) if args.expected else None
    print(f"{len(frames)} frames, {len(pipelines)} pipelines"
          + (f", expected {args.expected!r}" if args.expected else ""))

    stats = {name: collections.Counter() for name, _ in pipelines}
    reasons = {name: collections.Counter() for name, _ in pipelines}
    reads = {name: collections.Counter() for name, _ in pipelines}
    ok_conf = {name: [] for name, _ in pipelines}
    ms = {name: [] for name, _ in pipelines}
    per_variant = collections.defaultdict(collections.Counter)  # from the all_variants pipeline
    rows = []
    for i, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  skipped unreadable {path}")
            continue
        for name, settings in pipelines:
            for attr, value in settings.items():
                setattr(engine, attr, value)
            r = engine.read(frame)
            ms[name].append(r.get("processing_time_ms", 0.0))
            validation = r.get("validation")
            if r.get("error"):
                ok, reason = False, "ocr_engine_error"
            elif validation is not None:
                ok, reason = bool(validation["accepted"]), validation.get("reason")
            else:
                ok = bool(_norm(r.get("text")))
                reason = None if ok else "no_text_detected"
            text = r.get("text", "")
            if name == "all_variants" and validation:
                for v in validation["reads"]:
                    c = per_variant[v["variant"]]
                    c["n"] += 1
                    if v.get("accepted"):
                        c["ok" if expected is None or _norm(v.get("text")) == expected else "WRONG"] += 1
            reads[name][(("OK  " if ok else "NOK ") + text)] += 1
            if ok:
                hit = expected is None or _norm(text) == expected
                stats[name]["correct_ok" if hit else "WRONG_OK"] += 1
                if validation:
                    used = [x.get("min_token_confidence") for x in validation["reads"]
                            if x.get("accepted") and x.get("text") == text]
                    ok_conf[name] += [c for c in used if c is not None]
            else:
                stats[name]["rejected"] += 1
                reasons[name][reason] += 1
            rows.append({"frame": str(path), "pipeline": name, "status": "OK" if ok else "NOK",
                         "text": text, "reason": reason or "",
                         "correct": ("" if expected is None else int(ok and _norm(text) == expected))})
        if i % 10 == 0:
            print(f"  {i}/{len(frames)} frames done")

    n = len({r["frame"] for r in rows})
    print()
    for name, _ in pipelines:
        s = stats[name]
        if expected is not None:
            head = (f"correct OK {s['correct_ok']:4d}   WRONG OK {s['WRONG_OK']:4d}   "
                    f"rejected {s['rejected']:4d}   (of {n})")
        else:
            head = f"OK {s['correct_ok']:4d}   NOK {s['rejected']:4d}   (of {n})"
        print(f"{name:20s} {head}   ocr {np.mean(ms[name]):.0f} ms/frame")
        if reasons[name]:
            print("      reject reasons: " + ", ".join(f"{k}={v}" for k, v in reasons[name].most_common()))
        if ok_conf[name]:
            print(f"      min token confidence of accepted reads: lowest {min(ok_conf[name]):.3f}, "
                  f"5th percentile {np.percentile(ok_conf[name], 5):.3f}")
        for text, count in reads[name].most_common(4):
            print(f"      {count:4d}x  {text!r}")

    if per_variant:
        print("\nEach preprocessing variant on its own (validated single read):")
        print("  variant                accepted" + ("   of which WRONG" if expected else ""))
        for v, c in sorted(per_variant.items(), key=lambda kv: (-kv[1]["ok"], kv[1]["WRONG"])):
            print(f"  {v:22s} {c['ok'] + c['WRONG']:5d}/{c['n']}" + (f"   {c['WRONG']:5d}" if expected else ""))
        print("  Variants with WRONG > 0 are confidently misreading this print - keep them out of")
        print("  ocr.read_validation.consensus.variants.")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["frame", "pipeline", "status", "text", "reason", "correct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-frame results written to {args.csv}")

    try:
        engine._paddle.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
