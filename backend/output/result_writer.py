"""
Result Writer

Saves vision processing results to a daily JSONL file
(<output.result_dir>/<date>.jsonl, data/results by default).
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from backend.core.config_loader import cfg

log = logging.getLogger(__name__)

_write_lock = threading.Lock()

# Pre-resolve output directory at module load time
_result_dir = cfg.get("output", {}).get("result_dir", "data/results")
_output_dir = Path(_result_dir)
if not _output_dir.is_absolute():
    _output_dir = Path(__file__).resolve().parents[2] / _output_dir


_keep_days = int(cfg.get("output", {}).get("result_keep_days", 90) or 0)
_last_cleanup = None


def cleanup_old_results(keep_days=None, today=None):
    """Delete daily result files (<date>.jsonl) older than keep_days
    (output.result_keep_days; 0 = keep everything). Returns the number deleted."""
    keep = _keep_days if keep_days is None else int(keep_days)
    if keep <= 0 or not _output_dir.is_dir():
        return 0
    today = today or datetime.now().date()
    deleted = 0
    for f in _output_dir.glob("*.jsonl"):
        try:
            day = datetime.strptime(f.stem, "%Y-%m-%d").date()
        except ValueError:
            continue                                   # not a daily results file
        if (today - day).days > keep:
            try:
                f.unlink(); deleted += 1
            except OSError:
                log.warning("Could not delete old results file %s", f)
    if deleted:
        log.info("Deleted %d results file(s) older than %d days", deleted, keep)
    return deleted


def save_result(result: dict):
    """
    Appends a vision processing result to the daily JSONL file.
    """
    global _last_cleanup
    now = datetime.now()
    result["timestamp"] = now.isoformat()
    if _last_cleanup != now.date():                    # once per day, also right after startup
        _last_cleanup = now.date()
        cleanup_old_results()

    with _write_lock:
        try:
            _output_dir.mkdir(parents=True, exist_ok=True)
            daily_file = _output_dir / f"{now.date().isoformat()}.jsonl"
            with daily_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False))
                handle.write("\n")
        except Exception:
            log.exception("Failed to write result to JSONL file")