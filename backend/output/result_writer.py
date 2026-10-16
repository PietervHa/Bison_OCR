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


def save_result(result: dict):
    """
    Appends a vision processing result to the daily JSONL file.
    """
    now = datetime.now()
    result["timestamp"] = now.isoformat()

    with _write_lock:
        try:
            _output_dir.mkdir(parents=True, exist_ok=True)
            daily_file = _output_dir / f"{now.date().isoformat()}.jsonl"
            with daily_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False))
                handle.write("\n")
        except Exception:
            log.exception("Failed to write result to JSONL file")