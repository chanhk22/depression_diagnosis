#pose/gaze/AUs parsing

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFace 2.1.0 Pose/Gaze/AUs CSV -> cleaned time-series NPZ
- Robust delimiter detection
- Time column auto-detection (timestamp, frameTime, Time, time)
- Optional front cut (--t0_sec)
- Optional drop-intervals (--drop_intervals path) ex) CSV columns: start,end (sec)
- (선택) linear interpolation + forward/backward fill
- Outputs: {session_id}_openface.npz  with keys: feat (T,D), t (T,), meta (JSON str)

Example
-------
python features/openface_extract.py \
  --input /path/E-DAIC/Features/712_OpenFace2.1.0_Pose_gaze_AUs.csv \
  --outdir data/processed/openface \
  --session_id 712 \
  --t0_sec 3.2 \
  --drop_intervals data/processed/masks/712_silence.csv \
  --interpolate
"""

import os
import io
import json
import argparse
import numpy as np
import pandas as pd
import csv

# -----------------------------
# Helpers
# -----------------------------
CANDIDATE_TIME_COLS = ["timestamp", " timestamp", "frameTime", "time", "Time", "frame_time"]

def smart_read_csv(path: str) -> pd.DataFrame:
    """Read CSV with auto delimiter sniffing."""
    with open(path, "rb") as f:
        head = f.read(4096)
    try:
        sample = head.decode("utf-8", errors="ignore")
    except Exception:
        sample = head.decode(errors="ignore")

    sniffer = csv.Sniffer()
    dialect = None
    try:
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter
    except Exception:
        # default to comma
        delimiter = ","
    try:
        df = pd.read_csv(path, delimiter=delimiter)
    except Exception:
        # fallback tab
        df = pd.read_csv(path, delimiter="\t")
    return df

def find_time_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in CANDIDATE_TIME_COLS:
        if k in cols_lower:
            return cols_lower[k]
    # Sometimes OpenFace has "timestamp" only in index or no time; try frame/fps inference
    # If no time col, return None and we’ll create a synthetic time grid if 'frame' exists.
    return None

def load_drop_intervals(path: str):
    # Accept CSV (start,end) or JSON [{"start":..,"end":..}, ...]
    if path is None or not os.path.exists(path):
        return []
    if path.lower().endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        out = []
        for d in data:
            out.append((float(d["start"]), float(d["end"])))
        return out
    # CSV
    df = smart_read_csv(path)
    # try common names
    cand = [("start","end"), ("start_sec","end_sec"), ("s","e")]
    for a,b in cand:
        if a in df.columns and b in df.columns:
            return list(zip(df[a].astype(float).values, df[b].astype(float).values))
    # else assume first two numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        a = num_df.columns[0]; b = num_df.columns[1]
        return list(zip(num_df[a].astype(float).values, num_df[b].astype(float).values))
    return []

def apply_front_cut(t: np.ndarray, X: np.ndarray, t0: float):
    if t0 <= 0:
        return t, X
    keep = t >= t0
    return (t[keep] - t0), X[keep]

def apply_drop_intervals(t: np.ndarray, X: np.ndarray, intervals):
    if not intervals:
        return t, X
    keep_mask = np.ones_like(t, dtype=bool)
    for (s,e) in intervals:
        keep_mask &= ~((t >= s) & (t < e))
    return t[keep_mask], X[keep_mask]

def linear_interpolate_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.interpolate(method="linear", limit_direction="both", axis=0)
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="OpenFace Pose_gaze_AUs CSV path")
    ap.add_argument("--outdir", required=True, help="Output directory (npz)")
    ap.add_argument("--session_id", default=None, help="Override session id for output filename")
    ap.add_argument("--t0_sec", type=float, default=0.0, help="Front-cut seconds (Ellie removal)")
    ap.add_argument("--drop_intervals", type=str, default=None, help="CSV/JSON of (start,end) to drop")
    ap.add_argument("--interpolate", action="store_true", help="Interpolate NaNs")
    ap.add_argument("--keep_columns", nargs="*", default=None,
                    help="Explicit columns to keep (else auto-select numeric feature cols)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = smart_read_csv(args.input)

    # time
    time_col = find_time_column(df)
    if time_col is None:
        # Try to create synthetic time by frame index + fps≈30 / 33.333ms
        # OpenFace "frame" column often exists.
        fps = 30.0
        cand = [c for c in df.columns if c.lower() == "frame"]
        if cand:
            frame_col = cand[0]
            t = df[frame_col].astype(float).values / fps
        else:
            # fallback: 1/30 increments
            t = np.arange(len(df), dtype=float) / 30.0
    else:
        t = df[time_col].astype(float).values

    # feature columns
    if args.keep_columns is not None and len(args.keep_columns) > 0:
        feat_df = df[args.keep_columns]
    else:
        # Drop obviously non-feature columns
        drop_like = set([time_col, "frame", "timestamp", "frameTime", "success", "confidence"])
        drop_like |= set([c for c in df.columns if str(c).lower() in ["frame", "timestamp", "frametime", "success", "confidence"]])
        # Keep numeric columns only
        feat_df = df.drop(columns=[c for c in df.columns if c in drop_like and c in df.columns], errors="ignore")
        feat_df = feat_df.select_dtypes(include=[np.number])

    if args.interpolate:
        feat_df = linear_interpolate_nan(feat_df)

    X = feat_df.to_numpy(dtype=np.float32)

    # Front cut
    t, X = apply_front_cut(t, X, args.t0_sec)
    # Drop intervals
    intervals = load_drop_intervals(args.drop_intervals)
    t, X = apply_drop_intervals(t, X, intervals)

    # Re-base time to start at 0 (after cuts/drops)
    if len(t) > 0:
        t = t - (t.min() if np.isfinite(t.min()) else 0.0)

    # session id / filename
    if args.session_id is not None:
        sid = str(args.session_id)
    else:
        base = os.path.basename(args.input)
        sid = os.path.splitext(base)[0].split("_")[0]  # "712_OpenFace..." -> "712"

    out_path = os.path.join(args.outdir, f"{sid}_openface.npz")
    meta = {
        "source": os.path.abspath(args.input),
        "session_id": sid,
        "columns": list(feat_df.columns),
        "t0_sec": args.t0_sec,
        "drop_intervals": intervals,
        "interpolated": bool(args.interpolate),
        "num_frames": int(X.shape[0]),
        "feat_dim": int(X.shape[1]) if X.ndim == 2 else 0
    }
    np.savez_compressed(out_path, feat=X, t=t.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
    print(f"[OK] Saved: {out_path}  shape={X.shape}")

if __name__ == "__main__":
    main()
