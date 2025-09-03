# preprocessing/dvlog_visual_parser.py
import os
import numpy as np
from .clnf_parser import normalize_landmarks, interocular_distance

def load_dvlog_visual(path):
    """
    Load dvlog visual npy and convert to canonical (T,68,2) shape.
    dvlog may store:
      - a (T,136) array with interleaved [x0,y0,x1,y1,...] per frame
      - or a (T,68,2) array already
      - or an object array/list of per-frame pairs
    """
    a = np.load(path, allow_pickle=True)
    if isinstance(a, np.ndarray) and a.dtype == np.object_:
        # object array of per-frame lists -> convert
        arr = np.stack([np.array(x).reshape(-1) for x in a], axis=0)
        a = arr

    if a.ndim == 3 and a.shape[1] == 68 and a.shape[2] == 2:
        pts = a.astype(np.float32)
    elif a.ndim == 2 and a.shape[1] == 136:
        # interleaved x0,y0,x1,y1...
        T = a.shape[0]
        pts = a.reshape(T, 68, 2).astype(np.float32)
    elif a.ndim == 1 and a.dtype == np.object_:
        # a list of arrays per-frame
        frames = []
        for f in a:
            arr = np.asarray(f).reshape(-1)
            if arr.size == 136:
                frames.append(arr.reshape(68,2))
            else:
                raise RuntimeError("Unknown per-frame dimension in dvlog visual")
        pts = np.stack(frames, axis=0).astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected dvlog visual shape {a.shape} from {path}")

    return pts  # (T,68,2) float32

def normalize_dvlog_visual(path, method="interocular"):
    pts = load_dvlog_visual(path)
    pts_n = normalize_landmarks(pts, method=method)
    return pts_n
