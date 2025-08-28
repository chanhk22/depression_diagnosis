

# precache/window_cache.py
import os, math, numpy as np, pandas as pd
from ..preprocessing.utils_io import read_table_smart

def load_timeseries(csv_path, time_col_candidates=('frameTime','timestamp','timeStamp')):
    df, sep = read_table_smart(csv_path)
    tcol = [c for c in time_col_candidates if c in df.columns]
    if not tcol:
        return None, None, None  # ex) 세션 집약형이라면 None
    tcol = tcol[0]
    t = df[tcol].astype(float).values
    # feature columns = 나머지 수치형
    feats = df.select_dtypes(include=['number']).drop(columns=[tcol], errors='ignore')
    return t, feats.values, feats.columns.tolist()

def normalize_landmarks(pts, method="interocular"):
    """
    pts: (T,68,2)
    method: "interocular" (눈 거리로 정규화), "minmax"
    """
    if pts.ndim != 3 or pts.shape[1] != 68 or pts.shape[2] != 2:
        return pts
    if method == "interocular":
        left_eye = np.mean(pts[:,36:42,:], axis=1)
        right_eye = np.mean(pts[:,42:48,:], axis=1)
        dist = np.linalg.norm(left_eye - right_eye, axis=1, keepdims=True)
        dist[dist==0] = 1.0
        pts = pts / dist[:,None,:]
    elif method == "minmax":
        min_xy = pts.min(axis=1, keepdims=True)
        max_xy = pts.max(axis=1, keepdims=True)
        rng = (max_xy - min_xy); rng[rng==0] = 1.0
        pts = (pts - min_xy) / rng
    return pts

def process_landmarks(x, cols, source="clnf"):
    """
    CLNF: (x0..x67, y0..y67)
    dlib: (x0,y0,x1,y1,...,x67,y67)
    """
    if source == "clnf":
        pts = np.stack([x[:, :68], x[:, 68:]], axis=-1)  # (T,68,2)
    else:  # dlib
        pts = x.reshape(-1,68,2)
    pts = normalize_landmarks(pts, method="interocular")
    return pts

def build_windows(session_id, modal_csvs, win_len=4.0, stride=1.0, out_dir="./data/cache/EDAIC"):
    """
    modal_csvs: dict 예)
      {
        "audio": "path/to/egemaps_clean.csv",
        "mfcc_priv": "path/to/mfcc_clean.csv",
        "openface_priv": "path/to/openface_clean.csv",
        "vgg_priv": "path/to/vgg16_clean.csv",
        "dense_priv": "path/to/densenet201_clean.csv",
      }
    """
    os.makedirs(out_dir, exist_ok=True)

    tm = {}
    for k, p in modal_csvs.items():
        t, x, cols = load_timeseries(p)
        tm[k] = (t, x, cols)

    # 공통 시간축: audio 있으면 audio로, 없으면 longest
    if tm.get('audio') and tm['audio'][0] is not None:
        T = tm['audio'][0]
    else:
        # 가장 긴 타임라인
        T = max([v[0] for v in tm.values() if v[0] is not None], key=lambda a: a[-1])

    t_start, t_end = float(T[0]), float(T[-1])
    w = 0; index_rows = []
    cur = t_start
    while cur + win_len <= t_end:
        w_start, w_end = cur, cur + win_len
        window_dict = {}
        lengths = {}
        for k,(t,x,cols) in tm.items():
            if t is None: 
                continue
            sel = (t >= w_start) & (t < w_end)
            window_dict[k] = x[sel]
            lengths[k] = int(sel.sum())
        # 최소 요건: audio가 충분한 샘플 보유
        if 'audio' in window_dict and window_dict['audio'].shape[0] >= int(0.6*win_len*100):
            out_path = os.path.join(out_dir, f"{session_id}_w{w:05d}.npz")
            np.savez_compressed(out_path, **window_dict)
            index_rows.append({
                "session": session_id,
                "w": w,
                "t0": w_start,
                "t1": w_end,
                **{f"len_{k}": v for k,v in lengths.items()},
                "path": out_path
            })
            w += 1
        cur += stride

    idx = pd.DataFrame(index_rows)
    idx_path = os.path.join(out_dir, f"{session_id}_index.csv")
    idx.to_csv(idx_path, index=False)
    return idx_path
