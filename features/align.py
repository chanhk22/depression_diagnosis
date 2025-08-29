import numpy as np

def make_target_grid(start, end, hz):
    """
    Return times from start (inclusive) to end (exclusive) with frequency hz.
    """
    if end <= start:
        return np.array([], dtype=float)
    step = 1.0 / float(hz)
    n = int(np.floor((end - start) * hz))
    return start + np.arange(n) * step

def resample_to_target(source_t, source_x, target_t):
    """
    Linear interpolation per-dimension.
    source_t: (T_src,)
    source_x: (T_src, D) or (T_src,) or (T_src, d1, d2) (will be flattened on last dims)
    target_t: (T_tgt,)
    Returns: (T_tgt, D) or (T_tgt, d1, d2) matching flattened dims
    """
    if source_t is None or source_x is None or len(source_t)==0:
        # return zeros with correct dim if possible
        if isinstance(source_x, np.ndarray):
            shape = source_x.shape
            if source_x.ndim == 1:
                D = 1
                return np.zeros((len(target_t),), dtype=source_x.dtype)
            else:
                D = np.prod(shape[1:])
                return np.zeros((len(target_t), D), dtype=source_x.dtype)
        return None

    source_t = np.asarray(source_t, dtype=float)
    sx = np.asarray(source_x)
    # flatten trailing dims
    orig_shape = sx.shape
    if sx.ndim == 1:
        sx2 = sx[:, None]
    elif sx.ndim == 2:
        sx2 = sx
    else:
        sx2 = sx.reshape(sx.shape[0], -1)

    tgt = np.asarray(target_t, dtype=float)
    out = np.zeros((len(tgt), sx2.shape[1]), dtype=sx2.dtype)
    for d in range(sx2.shape[1]):
        col = sx2[:, d]
        # handle nan by interpolation on valid mask
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            out[:, d] = 0.0
            continue
        if mask.sum() < len(col):
            col = np.interp(source_t, source_t[mask], col[mask])
        out[:, d] = np.interp(tgt, source_t, col, left=col[0], right=col[-1])
    # restore shape
    if len(orig_shape) > 2:
        out = out.reshape(len(tgt), *orig_shape[1:])
    elif orig_shape == (len(source_t),):
        out = out.ravel()
    return out
