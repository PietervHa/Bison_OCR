"""
backend/utils/workers.py

A shared thread pool for the independent pieces of one OCRead read (the
line openings of dot_text, see ocr.speed.ink_threads). OpenCV releases the
GIL inside its calls, so they run in parallel on the other cores.

The workers run at the vision thread's priority
(performance.vision_thread_priority): a new thread starts at normal
priority, and the read waits for its slowest worker.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.core.config_loader import cfg
from backend.utils.priority import set_thread_priority

_pools: dict = {}
_lock = threading.Lock()


def _init_worker() -> None:
    set_thread_priority((cfg.get("performance") or {}).get("vision_thread_priority", "normal"))


def pool(n: int) -> ThreadPoolExecutor:
    """The shared pool with n workers, created on first use."""
    with _lock:
        if n not in _pools:
            _pools[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ocr_worker", initializer=_init_worker)
        return _pools[n]


def by_rows(fn, img: np.ndarray, overlap: int, n: int) -> np.ndarray:
    """fn(img) computed on n horizontal stripes in parallel, each with
    overlap rows of context above and below that are cut off again. The
    same result as fn(img) for a neighbourhood operation that sees at most
    overlap rows away (e.g. a morphological closing: twice its kernel
    radius). n <= 1: fn(img)."""
    H = img.shape[0]
    if n <= 1 or H < 2 * n:
        return fn(img)
    cuts = [round(H * i / n) for i in range(n + 1)]

    def stripe(i):
        a, b = cuts[i], cuts[i + 1]
        lo, hi = max(0, a - overlap), min(H, b + overlap)
        return fn(img[lo:hi])[a - lo:a - lo + (b - a)]
    return np.vstack(list(pool(n).map(stripe, range(n))))
