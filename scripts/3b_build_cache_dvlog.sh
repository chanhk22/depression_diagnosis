#!/usr/bin/env bash
set -e

python - <<'PY'
import os, yaml, glob, numpy as np, pandas as pd

with open("configs/default.yaml") as f:
    C = yaml.safe_load(f)

PROC   = C['outputs']['processed_root']
CACHE  = C['outputs']['cache_root']
L      = C['windowing']['length_s']
S      = C['windowing']['stride_s']
BASEHZ = C['windowing']['base_rate_hz']   # 오디오는 100Hz 기준
MINR   = C['windowing']['min_valid_ratio']

ACDIR  = C['processed']['dvlog']['acoustic']
VSDIR  = C['processed']['dvlog']['visual']
LBLCSV = C['labels']['dvlog']['labels_csv']

os.makedirs(f"{CACHE}/D-VLOG", exist_ok=True)

def build_one(session_id, ac_path, vs_path, out_dir):
    x_ac = np.load(ac_path)  # (T, D_audio)
    T = x_ac.shape[0]
    t_end = T / BASEHZ
    w, cur, rows = 0, 0.0, []
    x_vs = None
    if isinstance(vs_path, str) and os.path.exists(vs_path):
        x_vs = np.load(vs_path)   # (Tvis, 136) or (Tvis,68,2)
        # 간단 동기화: 시계열 길이를 시간 비율로 매핑해 슬라이스
        vis_hz = C['preprocessing']['resample']['vis_hz']
        def vis_slice(t0, t1):
            i0, i1 = int(t0*vis_hz), int(t1*vis_hz)
            i0 = max(i0, 0); i1 = min(i1, x_vs.shape[0])
            return x_vs[i0:i1]
    def ac_slice(t0, t1):
        i0, i1 = int(t0*BASEHZ), int(t1*BASEHZ)
        i0 = max(i0, 0); i1 = min(i1, x_ac.shape[0])
        return x_ac[i0:i1]

    while cur + L <= t_end + 1e-6:
        t0, t1 = cur, cur + L
        a = ac_slice(t0, t1)
        if a.shape[0] < int(MINR * L * BASEHZ):
            cur += S; continue
        save = {"audio": a}
        if x_vs is not None:
            v = vis_slice(t0, t1)
            if v.size > 0: save["landmarks"] = v
        outp = os.path.join(out_dir, f"{session_id}_w{w:05d}.npz")
        np.savez_compressed(outp, **save)
        rows.append({"session": session_id, "w": w, "t0": t0, "t1": t1, "path": outp,
                     "len_audio": int(a.shape[0]), "len_landmarks": int(save.get("landmarks", np.empty((0,))).shape[0])})
        w += 1; cur += S
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(out_dir, f"{session_id}_index.csv"), index=False)
    return idx

index_paths = []
for ac in glob.glob(os.path.join(ACDIR, "*.npy")):
    sid = os.path.splitext(os.path.basename(ac))[0]
    vs = os.path.join(VSDIR, f"{sid}.npy")
    idx = build_one(sid, ac, vs if os.path.exists(vs) else None, f"{CACHE}/D-VLOG")
    index_paths.append(idx)

if index_paths:
    all_df = pd.concat(index_paths, ignore_index=True)
    out_csv = os.path.join(f"{CACHE}/D-VLOG", "D-VLOG_all_index.csv")
    all_df.to_csv(out_csv, index=False)
    print(f"[3b_build_cache_dvlog] D-VLOG: {len(all_df)} windows -> {out_csv}")
else:
    print("[3b_build_cache_dvlog] no D-VLOG sessions found")
PY
