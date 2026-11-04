"""Daily results files older than output.result_keep_days are deleted."""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from backend.output import result_writer as rw


class ResultRetentionTests(unittest.TestCase):
    def test_old_files_deleted_recent_and_other_files_kept(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("2026-01-01.jsonl", "2026-10-20.jsonl", "2026-11-02.jsonl", "notes.jsonl"):
                (d / name).write_text("{}\n")
            with mock.patch.object(rw, "_output_dir", d):
                n = rw.cleanup_old_results(keep_days=30, today=date(2026, 11, 2))
            self.assertEqual(n, 1)
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["2026-10-20.jsonl", "2026-11-02.jsonl", "notes.jsonl"])

    def test_zero_keeps_everything(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "2020-01-01.jsonl").write_text("{}\n")
            with mock.patch.object(rw, "_output_dir", d):
                self.assertEqual(rw.cleanup_old_results(keep_days=0, today=date(2026, 11, 2)), 0)
            self.assertTrue((d / "2020-01-01.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
