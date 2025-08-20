# datasets/window.py
import numpy as np

def window_indices(T, win, hop):
    """Return list of (start, end) indices for windowing length-T sequence."""
    idx = []
    s = 0
    while s < T:
        e = min(T, s + win)
        idx.append((s, e))
        if e == T: break
        s += hop
    return idx

def apply_window(x, win, hop):
    """x: (T, D) -> (N_win, L, D) with last window zero-padded if needed."""
    T = x.shape[0]
    idxs = window_indices(T, win, hop)
    L = max(e - s for s, e in idxs)
    out = np.zeros((len(idxs), L, x.shape[1]), dtype=x.dtype)
    for i, (s, e) in enumerate(idxs):
        out[i, :e - s] = x[s:e]
    return out
