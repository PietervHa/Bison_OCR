"""Tests for the QC helpers: session report and project-relative paths."""
import json
import tempfile
import unittest
from pathlib import Path

from QC_tools import ocread_session_report as report


def _cycle(text, status="OK", reason=None, reads=(), t="2026-10-13 09:43:03", cycle_ms=3100.0):
    r = {"timestamp": t + ".000", "mode": "ocread", "status": status, "text": text,
         "cycle_time_ms": cycle_ms, "processing_time_ms": cycle_ms - 100}
    if reason:
        r["failure_reason"] = reason
    if reads:
        r["validation"] = {"accepted": status == "OK", "reason": reason,
                           "reads": [dict(variant=v, text=tx, reason=rs, min_token_confidence=c) for v, tx, rs, c in reads]}
    return r


class SessionReportTests(unittest.TestCase):
    def _write(self, rows):
        d = tempfile.mkdtemp()
        p = Path(d) / "2026-10-13.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    def test_counts_reasons_and_variants(self):
        good, bad = "4113 08:26 BB 05/26", "4113 08:25 BB 05/26"
        rows = [
            _cycle(good, reads=[("dot_divide", good, None, 0.95), ("dot_subtract", good, None, 0.91)]),
            _cycle(bad, reads=[("dot_divide", bad, None, 0.97), ("dot_subtract", bad, None, 0.96)], t="2026-10-13 09:44:00"),
            _cycle("4113 08:2 BB 05/26", status="NOK", reason="format_mismatch",
                   reads=[("dot_divide", None, "format_mismatch", None)], t="2026-10-13 09:45:00"),
            {"timestamp": "2026-10-13 09:46:00", "mode": "ocr", "status": "OK"},  # other mode: ignored
        ]
        out = report.summarize(report.load_cycles([self._write(rows)]), expected=good)
        self.assertIn("3 OCRead cycles", out)
        self.assertIn("OK 2   NOK 1", out)
        self.assertIn("format_mismatch 1", out)
        self.assertIn("correct OK 1   WRONG OK 1   rejected 1", out)
        self.assertIn("dot_divide", out)
        self.assertIn("'4113 08:2 BB 05/26'", out)

    def test_time_filter_and_old_results_without_validation(self):
        rows = [_cycle("4113 08:26 BB 05/26", t="2026-10-13 08:00:00"),
                _cycle("4113 08:26 BB 05/26", t="2026-10-13 10:00:00")]
        cycles = report.load_cycles([self._write(rows)], since="09:00")
        self.assertEqual(len(cycles), 1)
        self.assertIn("1 OCRead cycles", report.summarize(cycles))

    def test_missing_file_is_reported(self):
        self.assertEqual(report.main(["does/not/exist.jsonl"]), 1)


class ProjectPathTests(unittest.TestCase):
    def test_relative_debug_dir_is_project_root_relative(self):
        from backend.detection.ocread.engine import PROJECT_ROOT, resolve_project_path
        self.assertEqual(Path(resolve_project_path("data/debug/dynamic_roi")), PROJECT_ROOT / "data/debug/dynamic_roi")
        absolute = str(Path.cwd().resolve())
        self.assertEqual(resolve_project_path(absolute), absolute)

if __name__ == "__main__":
    unittest.main()
