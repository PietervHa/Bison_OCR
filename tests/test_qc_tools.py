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

class TrainFontReaderLabelTests(unittest.TestCase):
    def test_line1_candidates(self):
        from datetime import datetime
        from QC_tools.train_font_reader import capture_time, line1_candidates
        t = capture_time("UId_0_Time_1-10-2026_10-28-51_123.jpg")
        self.assertEqual(t, datetime(2026, 10, 1, 10, 28, 51))
        self.assertEqual(line1_candidates(t), [(-1, "6274 10:27"), (0, "6274 10:28"), (1, "6274 10:29")])

    def test_duplicates_are_removed(self):
        from datetime import datetime
        from QC_tools.train_font_reader import line1_candidates
        # +30 s stays in the same minute -> two candidates
        self.assertEqual(line1_candidates(datetime(2026, 10, 1, 10, 28, 10)),
                         [(-1, "6274 10:27"), (0, "6274 10:28")])

    def test_day_follows_the_shifted_time(self):
        from datetime import datetime
        from QC_tools.train_font_reader import line1_candidates
        self.assertEqual(line1_candidates(datetime(2026, 10, 1, 23, 59, 45))[-1], (1, "6275 00:00"))
        self.assertEqual(line1_candidates(datetime(2026, 10, 2, 0, 0, 20))[0], (-1, "6274 23:59"))

    def test_no_capture_time_in_the_name(self):
        from QC_tools.train_font_reader import capture_time
        self.assertIsNone(capture_time("photo.jpg"))


class TrainFontReaderLiteralCheckTests(unittest.TestCase):
    """literal_check: fixed characters must read as themselves, letters the
    model doesn't know must at least have print in their cell."""
    CLASSES = [str(d) for d in range(10)] + ["B", ":", "/"]

    def _check(self, template, text, ink=None):
        import numpy as np
        from types import SimpleNamespace
        from QC_tools.train_font_reader import literal_check
        reader = SimpleNamespace(idx={c: i for i, c in enumerate(self.CLASSES)})
        pos = [c for c in text if c != " "]
        p = np.full((len(pos), len(self.CLASSES)), 0.01)
        for i, c in enumerate(pos):
            if c in reader.idx:
                p[i, reader.idx[c]] = 0.9           # the model reads what is printed
        cells = [np.full((24, 16), (ink or {}).get(i, 0.2), np.float32) for i in range(len(pos))]
        return literal_check(reader, template, cells, p)

    def test_matching_print_passes(self):
        self.assertIsNone(self._check(["#### ##:##", "BB ##/##"], "6274 09:16 BB 05/28"))

    def test_grid_shifted_by_one_fails_on_colon(self):
        self.assertEqual(self._check(["#### ##:##"], "274 09:16 2"), "':' not read at its position")

    def test_b_must_read_as_b(self):
        self.assertEqual(self._check(["#### ##:##", "BB ##/##"], "6274 09:16 EX 05/28"), "'B' not read at its position")

    def test_unknown_letter_on_an_empty_cell_fails(self):
        # EXP grid on a BB print: the P lands on the gap before the date
        self.assertEqual(self._check(["#### ##:##", "EXP ##/##"], "6274 09:16 BBP 05/28", ink={11: 0.01}),
                         "no print at letter 'P'")
        self.assertIsNone(self._check(["#### ##:##", "EXP ##/##"], "6274 09:16 EXP 05/28"))


class TrainFontReaderSegmentFormatsTests(unittest.TestCase):
    """segment_formats (shared work across formats) gives exactly what
    font_reader.segment() gives per format, orientation and angle search."""

    def test_same_as_segment_per_format(self):
        import cv2
        from backend.detection.ocr.font_reader import ink_maps, segment
        from QC_tools.train_font_reader import segment_formats
        fix = Path(__file__).resolve().parent / "fixtures" / "font_reader"
        formats = [["#### ##:##", "BB ##/##"], ["#### ##:##", "EXP ##/##"], ["#### ##:##", "PROD ##/##"], ["#### ##:##"]]
        for name in ("4113_upside_down.png", "3030_crossing_ribs.png"):
            _, raw = ink_maps(cv2.imread(str(fix / name)))
            got = list(segment_formats(raw, formats))
            self.assertGreaterEqual(len(got), 2 * len(formats))
            for tpl, (flip, first_deg), seg in got:
                with self.subTest(name=name, tpl=tpl, flip=flip, first_deg=first_deg):
                    self.assertEqual(seg, segment(raw[::-1, ::-1].copy() if flip else raw, tpl, first_deg=first_deg))


if __name__ == "__main__":
    unittest.main()
