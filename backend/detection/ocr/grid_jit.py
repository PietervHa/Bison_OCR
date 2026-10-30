"""
backend/detection/ocr/grid_jit.py

The cell grid search of font_reader._fit_cells compiled with numba
(optional dependency, ocr.speed.grid_jit): every pitch and offset at full
resolution - the exhaustive search, with exactly its grids and scores (the
same floating point operations in the same order, the first maximum in
(layout, pitch, offset) order) - instead of the numpy coarse-to-fine
search. When numba is not installed or the kernel does not compile,
available() is False and font_reader uses the numpy search.

The compiled kernel is cached on disk (numba cache=True, __pycache__ next
to this file): only the very first start compiles (~2 s).
"""

from __future__ import annotations

import numpy as np

from backend.utils.logger import get_logger

log = get_logger(__name__)

_status = None          # None: not tried yet; else (available, description)

try:
    import numba

    @numba.njit(cache=True, inline="always")
    def _mean(T, W, a, b):
        """Mean of the column profile over [rint(a), rint(b)) clamped to
        [0, W]: T[b - a, a] (see _means)."""
        ia = int(np.rint(a)); ib = int(np.rint(b))
        ia = 0 if ia < 0 else (W if ia > W else ia)
        ib = 0 if ib < 0 else (W if ib > W else ib)
        return T[ib - ia, ia]

    @numba.njit(cache=True)
    def _means(cs, W, dmax):
        """T[d, a] = (cs[a + d] - cs[a]) / max(d, 1) for every window up to
        dmax px: the value _GridTerms.mean computes, divided once per window
        instead of once per use (the division was most of the search)."""
        T = np.zeros((dmax + 1, W + 1))
        for d in range(dmax + 1):
            for a in range(W + 1 - d):
                T[d, a] = (cs[a + d] - cs[a]) / (d if d > 1 else 1)
        return T

    @numba.njit(cache=True)
    def _search(cs, W, ps, x0, kinds, ks, lens):
        """Best grid per layout l: (score[l], pitch row[l], offset column[l]),
        -inf / -1 when layout l has no offsets. kinds[l, k]: 1 = glyph cell,
        0 = space; ks[l]: cells of layout l; lens[l, r]: offsets of layout l
        at pitch row r (a prefix of the row of x0)."""
        L, kmax = kinds.shape
        # The widest window is a margin (one pitch) or a core (0.7 pitch);
        # +2 for the rounding of both ends.
        T = _means(cs, W, min(W, int(np.ceil(ps.max())) + 2))
        best = np.full(L, -np.inf); br = np.full(L, -1); bi = np.full(L, -1)
        core = np.empty(kmax); edge = np.empty(kmax)
        for r in range(ps.shape[0]):
            p = ps[r]
            n = 0
            for l in range(L):
                n = max(n, lens[l, r])
            for i in range(n):
                x = x0[r, i]
                for k in range(kmax):
                    a = x + k * p
                    core[k] = _mean(T, W, a + 0.15 * p, a + 0.85 * p)
                    edge[k] = _mean(T, W, a - 0.08 * p, a + 0.08 * p)
                left = _mean(T, W, x - p, x)
                for l in range(L):
                    if i >= lens[l, r]:
                        continue
                    K = ks[l]
                    s = 0.0
                    for k in range(K):
                        if kinds[l, k] == 1:
                            s += core[k]
                        else:
                            s += -2.0 * core[k]
                        s -= 0.5 * edge[k]
                    s -= 2.0 * (left + _mean(T, W, x + K * p, x + (K + 1) * p))
                    s /= K
                    if s > best[l]:
                        best[l] = s; br[l] = r; bi[l] = i
        return best, br, bi

    _import_error = None
except Exception as exc:                        # not installed, or broken install
    _search = None
    _import_error = exc


def available() -> bool:
    """True when the compiled search can be used: numba imports and the
    kernel compiles (or loads from the cache) and runs on a small case.
    Tried once; the outcome is logged."""
    global _status
    if _status is None:
        if _search is None:
            _status = (False, f"numba not available ({_import_error})")
        else:
            try:
                cs = np.r_[0.0, np.cumsum(np.ones(40))]
                _search(cs, 40, np.array([10.0]), np.zeros((1, 4)), np.ones((1, 2), np.int64),
                        np.array([2], np.int64), np.array([[4]], np.int64))
                _status = (True, f"numba {numba.__version__}")
            except Exception as exc:
                _status = (False, f"numba kernel failed to compile ({exc})")
        log.info("Cell grid search: %s", "compiled, " + _status[1] if _status[0]
                 else "numpy coarse-to-fine - " + _status[1])
    return _status[0]


def status() -> str:
    """'numba <version>' or why the numpy search is used."""
    available()
    return _status[1]


def best_grids(cs, W, ps, x0, layouts, lens):
    """The compiled exhaustive search for _fit_cells: per layout (cells,
    labels) its best (score, pitch row, offset column), in layout order."""
    kmax = max(len(cells) for cells, _ in layouts)
    kinds = np.zeros((len(layouts), kmax), np.int64)
    for l, (cells, _) in enumerate(layouts):
        kinds[l, :len(cells)] = [1 if c == "G" else 0 for c in cells]
    ks = np.array([len(cells) for cells, _ in layouts], np.int64)
    best, br, bi = _search(cs, int(W), ps, x0, kinds, ks, np.asarray(lens, np.int64))
    return [(float(best[l]), int(br[l]), int(bi[l])) for l in range(len(layouts))]
