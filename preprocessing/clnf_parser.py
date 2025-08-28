# preprocessing/clnf_parser.py
import numpy as np
import re
import pandas as pd
from .utils_io import read_table_smart
import os

IBUG_LEFT_EYE  = [36,37,38,39,40,41]
IBUG_RIGHT_EYE = [42,43,44,45,46,47]
IBUG_MOUTH = [60,61,62,63,64,65,66,67]

def read_clnf_features_txt(path):
    """
    Parse CLNF features text/CSV where layout is:
      frame, timestamp, confidence, detection_success, x0, x1, ..., x67, y0, y1, ..., y67
    Returns:
      times: (T,) or None
      pts: (T,68,2) float32
    """
    df, sep = read_table_smart(path)
    cols = list(df.columns)

    # find time column if present
    time_col = None
    for cand in ["timestamp","time","frameTime","timeStamp","frame"]:
        for c in cols:
            if c.lower() == cand.lower():
                time_col = c
                break
        if time_col is not None:
            break

    # Try to locate x0..x67 / y0..y67 blocks by header names
    x_cols = [c for c in cols if re.match(r'^(x|X)\d+$', str(c))]
    y_cols = [c for c in cols if re.match(r'^(y|Y)\d+$', str(c))]

    if len(x_cols) == 68 and len(y_cols) == 68:
        # Ensure ordering x0..x67 and y0..y67
        x_cols_sorted = sorted(x_cols, key=lambda s: int(re.findall(r'\d+', s)[0]))
        y_cols_sorted = sorted(y_cols, key=lambda s: int(re.findall(r'\d+', s)[0]))
        xs = df[x_cols_sorted].to_numpy(dtype=np.float32)
        ys = df[y_cols_sorted].to_numpy(dtype=np.float32)
    else:
        # fallback: heuristic contiguous numeric block containing 136 numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 136:
            # find contiguous block of length 136 with non-trivial variance
            found = False
            for i in range(len(numeric_cols)-135):
                block = numeric_cols[i:i+136]
                arr = df[block].to_numpy(dtype=np.float32)
                if np.nanstd(arr) > 1e-6:
                    xs = arr[:, :68]
                    ys = arr[:, 68:136]
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Cannot locate CLNF landmark numeric block in {path}")
        else:
            raise RuntimeError(f"Not enough numeric columns to parse CLNF landmarks in {path}")

    T = xs.shape[0]
    pts = np.stack([xs, ys], axis=-1)   # (T,68,2) but currently xs is (T,68) -> stack -> (T,68,2) OK

    times = None
    if time_col is not None:
        try:
            times = df[time_col].astype(float).to_numpy()
        except Exception:
            times = np.arange(T, dtype=np.float32) / 30.0

    return times, pts.astype(np.float32)

def interocular_distance(pts):
    """
    pts: (T,68,2) normalized/unnormalized
    returns: (T,) interocular (distance between eye centers)
    """
    left_center = pts[:, IBUG_LEFT_EYE, :].mean(axis=1)  # (T,2)
    right_center = pts[:, IBUG_RIGHT_EYE, :].mean(axis=1)
    dist = np.linalg.norm(right_center - left_center, axis=1)
    return dist

def normalize_landmarks(pts, method="interocular"):
    """
    Normalize landmarks to make CLNF and dlib comparable.
    - method="interocular": center by mean, divide by interocular distance
    - method="zscore": per-coordinate z-score normalization (over time)
    - method="none": no change
    Input: pts (T,68,2)
    Output: pts_norm (T,68,2) float32
    """
    pts = pts.astype(np.float32)
    if method == "none":
        return pts
    if method == "interocular":
        # center on midpoint between eyes
        left = pts[:, IBUG_LEFT_EYE, :].mean(axis=1, keepdims=True)  # (T,1,2)
        right = pts[:, IBUG_RIGHT_EYE, :].mean(axis=1, keepdims=True)
        center = (left + right) / 2.0
        centered = pts - center  # (T,68,2)
        iod = np.linalg.norm(right - left, axis=2).squeeze(-1) if right.shape[-1]==2 else interocular_distance(pts)
        # iod shape (T,), avoid zero
        iod = interocular_distance(pts)
        iod = np.maximum(iod, 1e-6).reshape(-1,1,1)
        normed = centered / iod
        return normed.astype(np.float32)
    if method == "zscore":
        # compute mean and std per coordinate across time
        mean = np.nanmean(pts, axis=0, keepdims=True)  # (1,68,2)
        std  = np.nanstd(pts, axis=0, keepdims=True) + 1e-6
        return ((pts - mean) / std).astype(np.float32)
    raise ValueError(f"Unknown normalize method: {method}")

def ear_mar(pts):
    def eye_ratio(p, idx):
        v1=np.linalg.norm(p[:,idx[1]]-p[:,idx[5]],axis=1)
        v2=np.linalg.norm(p[:,idx[2]]-p[:,idx[4]],axis=1)
        h =np.linalg.norm(p[:,idx[0]]-p[:,idx[3]],axis=1)+1e-6
        return (v1+v2)/(2*h)
    EAR=(eye_ratio(pts,IBUG_LEFT_EYE)+eye_ratio(pts, IBUG_RIGHT_EYE))/2
    p=pts[:,IBUG_MOUTH,:]
    MAR=np.linalg.norm(p[:,2]-p[:,6],axis=1)/(np.linalg.norm(p[:,0]-p[:,4],axis=1)+1e-6)
    return EAR, MAR