import os, math, glob, numpy as np, pandas as pd
from datetime import datetime

# try to import helper from repo; if not, provide small fallback
try:
    from preprocessing.utils_io import read_table_smart
except Exception:
    import pandas as pd
    def read_table_smart(path):
        df = pd.read_csv(path, engine="python", error_bad_lines=False)
        return df, ','

from features.align import make_target_grid, resample_to_target
from preprocessing.clnf_parser import normalize_landmarks

def load_timeseries(csv_path, time_col_candidates=('frameTime','timestamp','timeStamp','start_time','time')):
    df, sep = read_table_smart(csv_path)
    tcol = next((c for c in time_col_candidates if c in df.columns), None)
    if tcol is None:
        return None, None, None
    t = df[tcol].astype(float).values
    feats = df.select_dtypes(include=['number']).drop(columns=[tcol], errors='ignore')
    return t, feats.values, feats.columns.tolist()

def detect_landmark_format(cols):
    # cols: list of column names
    # If format contains x0..x67 then y0..y67 -> clnf
    xs = [c for c in cols if str(c).lower().startswith('x')]
    ys = [c for c in cols if str(c).lower().startswith('y')]
    if len(xs) >= 68 and len(ys) >= 68:
        # heuristic: first 68 are x0..x67 and next 68 are y0..y67
        # if col sequence contains x0,x1.. then likely clnf
        return "clnf"
    # else fallback to dlib-like interleaved
    return "dlib"

def process_landmarks_block(x_block, cols=None):
    """
    x_block: numpy array (T, D) where D==136
    detect whether clnf or interleaved dlib and convert to (T,68,2)
    """
    if x_block is None:
        return None
    D = x_block.shape[1]
    if D != 136:
        # try reshape if already (T,68,2)
        if x_block.ndim == 3 and x_block.shape[1]==68 and x_block.shape[2]==2:
            return x_block.astype(np.float32)
        raise RuntimeError(f"Unexpected landmark dimension {D}")
    # Heuristic detection by column names was done earlier; fallback: check variance pattern
    # Assume CLNF if first half and second half have different value ranges (x ~ 0..W, y ~ 0..H)
    xs = x_block[:, :68]
    ys = x_block[:, 68:136]
    # if typical clnf, xs mean >> ys mean? Not always. We choose column-name detection earlier if possible.
    # We'll attempt both: create clnf and interleaved, check which yields smaller bounding-box height/width ratio typical.
    pts_clnf = np.stack([xs, ys], axis=-1)  # (T,68,2)
    pts_dlib = x_block.reshape(x_block.shape[0], 68, 2)
    # compute interocular dist std as heuristic
    iod_clnf = np.linalg.norm(pts_clnf[:,36:42,:].mean(1) - pts_clnf[:,42:48,:].mean(1), axis=1).mean()
    iod_dlib = np.linalg.norm(pts_dlib[:,36:42,:].mean(1) - pts_dlib[:,42:48,:].mean(1), axis=1).mean()
    # choose the one with larger, but non-zero iod (just heuristic)
    if iod_clnf > iod_dlib:
        return pts_clnf.astype(np.float32)
    else:
        return pts_dlib.astype(np.float32)

def build_windows(session_id, modal_csvs, win_len=4.0, stride=1.0, base_hz=100, out_dir="./data/cache/EDAIC", min_audio_ratio=0.6, normalize_landmarks_method="interocular"):
    """
    modal_csvs: dict of modality -> path OR for landmarks: ("path", "clnf" or "dlib" or None)
    Example:
      modal_csvs = {
         "audio": "/.../300_OpenSMILE2.3.0_egemaps.csv",
         "mfcc": "/.../300_OpenSMILE2.3.0_mfcc.csv",
         "landmarks": ("/.../300_OpenFace2.1.0_Pose_gaze_AUs.csv", "clnf"),
         "vgg": "/.../300_vgg16.csv"
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    # load all time series
    tm = {}
    for k,p in modal_csvs.items():
        if p is None:
            tm[k] = (None, None, None); continue
        if k == "landmarks":
            # p may be (path, source_hint) or path
            if isinstance(p, (list, tuple)):
                path, hint = p[0], p[1] if len(p)>1 else None
            else:
                path, hint = p, None
            t, x, cols = load_timeseries(path)
            if x is None:
                tm[k] = (t, None, None); continue
            # detect format by columns if possible
            fmt = None
            if cols:
                fmt = "clnf" if any(str(c).lower().startswith('x') for c in cols) else None
            pts = process_landmarks_block(x, cols)
            # normalize now or later
            if normalize_landmarks_method:
                pts = normalize_landmarks(pts, method=normalize_landmarks_method)
            tm[k] = (t, pts, cols)
        else:
            t, x, cols = load_timeseries(p)
            tm[k] = (t, x, cols)

    # choose base timeline (audio preferred)
    base_t = None
    if tm.get("audio") and tm["audio"][0] is not None:
        base_t = tm["audio"][0]
    else:
        # choose modality with smallest median dt (highest sample rate)
        cand = [(k, v[0]) for k,v in tm.items() if v[0] is not None]
        if not cand:
            raise RuntimeError("No modality has timestamps")
        best = min(cand, key=lambda kv: np.median(np.diff(kv[1])))
        base_t = best[1]

    t_start, t_end = float(base_t[0]), float(base_t[-1])
    index_rows = []
    w = 0
    cur = t_start
    while cur + win_len <= t_end:
        w0, w1 = cur, cur + win_len
        # build target grid for this window
        target_t = make_target_grid(w0, w1, base_hz)
        window_payload = {}
        lengths = {}
        for k, (t, x, cols) in tm.items():
            if t is None or x is None:
                window_payload[k] = None
                lengths[k] = 0
                continue
            # resample to target grid using align.resample_to_target
            y = resample_to_target(t, x, target_t)
            window_payload[k] = y
            # store length in frames
            lengths[k] = 0 if y is None else (0 if np.asarray(y).size==0 else y.shape[0])
        # audio presence check
        if window_payload.get("audio") is None or window_payload["audio"].shape[0] < int(min_audio_ratio * win_len * base_hz):
            cur += stride; continue

        out_path = os.path.join(out_dir, f"{session_id}_w{w:05d}.npz")
        # save only non-None modalities
        save_dict = {}
        for k,v in window_payload.items():
            if v is None:
                continue
            # store numpy arrays as 'audio','landmarks','mfcc', etc.
            save_dict[k] = v
        np.savez_compressed(out_path, **save_dict)

        index_rows.append({
            "session": session_id,
            "w": w,
            "t0": w0,
            "t1": w1,
            **{f"len_{k}": int(lengths[k]) for k in lengths},
            "path": out_path
        })
        w += 1
        cur += stride

    idx = pd.DataFrame(index_rows)
    idx_path = os.path.join(out_dir, f"{session_id}_index.csv")
    idx.to_csv(idx_path, index=False)
    return idx_path
