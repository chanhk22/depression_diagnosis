# eGeMAPS vs MFCC 등 동기화 헬퍼

# features/align.py
"""
Temporal alignment helpers: resample feature time-series to a target time grid.
"""
import numpy as np

def resample_to_target(source_t, source_x, target_t):
    """
    source_t: (T_src,) in seconds
    source_x: (T_src, D) numeric
    target_t: (T_tgt,) desired times
    returns: (T_tgt, D) via linear interpolation (per-dim)
    """
    source_t = np.asarray(source_t, dtype=float)
    source_x = np.asarray(source_x, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    if source_x.ndim == 1:
        source_x = source_x[:, None]
    T_src, D = source_x.shape
    out = np.zeros((target_t.shape[0], D), dtype=source_x.dtype)
    for d in range(D):
        col = source_x[:, d]
        # handle nan by filling
        if np.isnan(col).any():
            # simple forward/backward fill
            mask = ~np.isnan(col)
            if mask.sum() == 0:
                col = np.zeros_like(col)
            else:
                col = np.interp(source_t, source_t[mask], col[mask])
        out[:, d] = np.interp(target_t, source_t, col, left=col[0], right=col[-1])
    return out

def make_target_grid(start, end, hz):
    """
    returns numpy array of times from start to end (inclusive start, exclusive end)
    """
    if end <= start:
        return np.array([], dtype=float)
    step = 1.0 / float(hz)
    n = int(np.floor((end - start) * hz))
    return np.arange(n) * step + start
