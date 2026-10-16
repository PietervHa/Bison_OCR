"""
Report on what OCRead actually did during a session on the line, straight
from the results file the app already writes (data/results/<date>.jsonl).
Nothing is re-processed and no OCR engine is started - runs in seconds.

Shows:
  - OK / NOK counts and the reject reasons
  - cycle and processing times (median / 90th percentile / max)
  - with --expected: correct OK / WRONG OK / rejected
  - for each preprocessing variant (read validation): how often its read fit
    the format, why it didn't, and how often it agreed with the final code
  - the confidence of reads that agreed with the final code, to set
    ocr.read_validation.min_confidence
  - the most common reads of rejected bottles

Usage (from the repo root):
    python -m QC_tools.ocread_session_report                         (today)
    python -m QC_tools.ocread_session_report --since 09:40 --expected "4113 08:26 BB 05/26"
    python -m QC_tools.ocread_session_report data/results/2026-10-13.jsonl --last 50
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path


def _norm(text) -> str:
    return "".join(str(text or "").split())


def _pct(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]


def _results_dir() -> Path:
    try:
        from backend.output.result_writer import _output_dir
        return Path(_output_dir)
    except Exception:
        return Path(__file__).resolve().parents[1] / "data" / "results"


def load_cycles(paths, since=None, until=None, last=0):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("mode") != "ocread":
                    continue
                t = str(r.get("timestamp", ""))
                hhmm = t[11:16] if len(t) >= 16 else ""
                if since and hhmm and hhmm < since:
                    continue
                if until and hhmm and hhmm > until:
                    continue
                rows.append(r)
    return rows[-last:] if last else rows


def summarize(rows, expected=None) -> str:
    out = []
    p = out.append
    if not rows:
        return "No OCRead cycles found."
    exp = _norm(expected) if expected else None
    ts = [str(r.get("timestamp", "")) for r in rows]
    p(f"{len(rows)} OCRead cycles  ({ts[0][:19]} .. {ts[-1][:19]})")

    status = collections.Counter(r.get("status") for r in rows)
    p(f"Status: OK {status.get('OK', 0)}   NOK {status.get('NOK', 0)}")
    reasons = collections.Counter(r.get("failure_reason") or "?" for r in rows if r.get("status") != "OK")
    if reasons:
        p("NOK reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    if exp is not None:
        ok = [r for r in rows if r.get("status") == "OK"]
        right = sum(_norm(r.get("text")) == exp for r in ok)
        p(f"Against expected {expected!r}: correct OK {right}   WRONG OK {len(ok) - right}   "
          f"rejected {len(rows) - len(ok)}")

    for key, label in (("cycle_time_ms", "cycle time"), ("processing_time_ms", "OCR time")):
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            p(f"{label:10s} median {_pct(vals, 50):7.0f} ms   p90 {_pct(vals, 90):7.0f} ms   max {max(vals):7.0f} ms")

    with_val = [r for r in rows if isinstance(r.get("validation"), dict) and r["validation"].get("reads")]
    if with_val:
        p("")
        p(f"Per variant ({len(with_val)} cycles with read-validation data):")
        p("  variant                 fit format  agreed w/ final*  rejected for")
        stats = collections.defaultdict(lambda: collections.Counter())
        agree_conf = []
        for r in with_val:
            final = _norm(r.get("text")) if r.get("status") == "OK" else None
            target = exp or final
            for v in r["validation"]["reads"]:
                s = stats[v.get("variant") or "default"]
                s["n"] += 1
                if v.get("text"):
                    s["fit"] += 1
                    if target and _norm(v["text"]) == target:
                        s["agree"] += 1
                        if v.get("min_token_confidence") is not None:
                            agree_conf.append(float(v["min_token_confidence"]))
                    elif target:
                        s["disagree"] += 1
                if v.get("reason"):
                    s["reason:" + v["reason"]] += 1
        for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["agree"]):
            why = ", ".join(f"{k[7:]} {c}" for k, c in s.most_common() if k.startswith("reason:"))
            p(f"  {name:22s} {s['fit']:4d}/{s['n']:<4d}  {s['agree']:4d} (other code {s['disagree']})  {why}")
        p("  * agreed with " + ("the expected code" if exp else "the code finally reported OK"))
        if agree_conf:
            p(f"Confidence of reads that agreed: min {min(agree_conf):.3f}   p5 {_pct(agree_conf, 5):.3f}   "
              f"median {_pct(agree_conf, 50):.3f}")
            p("  -> ocr.read_validation.min_confidence should sit below the p5 value.")

    nok_reads = collections.Counter(r.get("text") or "" for r in rows if r.get("status") != "OK")
    if nok_reads:
        p("")
        p("Most common reads of rejected bottles:")
        for text, n in nok_reads.most_common(8):
            p(f"  {n:4d}x  {text!r}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="results .jsonl file(s); default: today's")
    ap.add_argument("--since", help="only cycles from HH:MM")
    ap.add_argument("--until", help="only cycles up to HH:MM")
    ap.add_argument("--last", type=int, default=0, help="only the last N cycles")
    ap.add_argument("--expected", help='expected code, e.g. "4113 08:26 BB 05/26"')
    ap.add_argument("--config", default=None, help="config YAML (read by backend.core.config_loader)")
    args = ap.parse_args(argv)

    files = args.files or [_results_dir() / f"{dt.date.today():%Y-%m-%d}.jsonl"]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print("Results file not found: " + ", ".join(str(m) for m in missing))
        print(f"Results are written to {_results_dir()} - pass a file from there.")
        return 1
    print(summarize(load_cycles(files, args.since, args.until, args.last), args.expected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
