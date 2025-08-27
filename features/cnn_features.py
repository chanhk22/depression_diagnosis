#VGG16/DenseNet201/ResNet feature loader

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN feature CSV (VGG16 or DenseNet201) -> cleaned time-series NPZ
- Works with E-DAIC: 712_vgg16.csv, 712_densenet201.csv, or CNN_*.mat converted to CSV
- Auto-detect time column
- Optional front cut / drop intervals
- Saves: {session_id}_{backbone}.npz with keys: feat (T,D), t (T,), meta (JSON)

Example
-------
python features/cnn_features.py \
  --input /path/E-DAIC/Features/712_vgg16.csv \
  --outdir data/processed/cnn \
  --backbone vgg16 \
  --session_id 712 \
  --t0_sec 3.2 \
  --drop_intervals data/processed/masks/712_silence.csv
"""

import os
import io
import json
import argparse
import numpy as np
import pandas as pd

CANDIDATE_TIME_COLS = ["timestamp", "frameTime", "time", "Time", "frame_time", "sec", "seconds"]

def smart_read_csv(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.read(4096)
    try:
        sample = head.decode("utf-8", errors="ignore")
    except Exception:
        sample = head.decode(errors="ignore")

    import csv
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","
    try:
        df = pd.read_csv(path, delimiter=delimiter)
    except Exception:
        df = pd.read_csv(path, delimiter="\t")
    return df

def find_time_column(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in CANDIDATE_TIME_COLS:
        if k in low:
            return low[k]
    # Many CNN csvs have "frame" + 1 Hz sampling; create synthetic time if needed
    return None

def load_drop_intervals(path: str):
    if path is None or not os.path.exists(path):
        return []
    if path.lower().endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        return [(float(d["start"]), float(d["end"])) for d in data]
    df = smart_read_csv(path)
    # heuristic for start/end columns
    for a,b in [("start","end"), ("start_sec","end_sec"), ("s","e")]:
        if a in df.columns and b in df.columns:
            return list(zip(df[a].astype(float).values, df[b].astype(float).values))
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

def choose_feature_cols(df: pd.DataFrame, time_col: str, explicit_cols=None):
    if explicit_cols:
        return df[explicit_cols]
    # Drop obvious non-feature cols
    drops = set([time_col, "frame", "Frame", "index", "filename"])
    cand = df.drop(columns=[c for c in df.columns if c in drops and c in df.columns], errors="ignore")
    # CNN features are numeric, high-dim (e.g., 4096 for VGG16, 2048 for DenseNet201)
    cand = cand.select_dtypes(include=[np.number])
    # Some E-DAIC files include a redundant "id" or "clip" numeric; keep all numeric for safety.
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="VGG16/DenseNet201 CSV path")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backbone", required=True, choices=["vgg16", "densenet201", "resnet", "cnn"])
    ap.add_argument("--session_id", default=None)
    ap.add_argument("--t0_sec", type=float, default=0.0)
    ap.add_argument("--drop_intervals", type=str, default=None)
    ap.add_argument("--keep_columns", nargs="*", default=None,
                    help="Explicit feature columns if you want to subset")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = smart_read_csv(args.input)
    time_col = find_time_column(df)
    if time_col is None:
        # create synthetic time @ 1 Hz (most CNN csvs are per-second embeddings)
        t = np.arange(len(df), dtype=float)
    else:
        t = df[time_col].astype(float).values

    feat_df = choose_feature_cols(df, time_col, args.keep_columns)
    X = feat_df.to_numpy(dtype=np.float32)

    # Front cut / drop
    t, X = apply_front_cut(t, X, args.t0_sec)
    intervals = load_drop_intervals(args.drop_intervals)
    t, X = apply_drop_intervals(t, X, intervals)
    if len(t) > 0:
        t = t - (t.min() if np.isfinite(t.min()) else 0.0)

    # session id
    if args.session_id is not None:
        sid = str(args.session_id)
    else:
        base = os.path.basename(args.input)
        sid = os.path.splitext(base)[0].split("_")[0]

    out_path = os.path.join(args.outdir, f"{sid}_{args.backbone}.npz")
    meta = {
        "source": os.path.abspath(args.input),
        "session_id": sid,
        "backbone": args.backbone,
        "columns": list(feat_df.columns),
        "t0_sec": args.t0_sec,
        "drop_intervals": intervals,
        "num_frames": int(X.shape[0]),
        "feat_dim": int(X.shape[1]) if X.ndim == 2 else 0
    }
    np.savez_compressed(out_path, feat=X, t=t.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
    print(f"[OK] Saved: {out_path}  shape={X.shape}")

if __name__ == "__main__":
    main()
