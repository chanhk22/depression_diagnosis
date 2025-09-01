#!/usr/bin/env bash
set -e

python - <<'PY'
import os, yaml, glob, pandas as pd
from precache.window_cache import build_windows

with open("configs/default.yaml") as f:
    C = yaml.safe_load(f)

PROC  = C['outputs']['processed_root']
CACHE = C['outputs']['cache_root']
L     = C['windowing']['length_s']
S     = C['windowing']['stride_s']
R     = C['windowing']['min_valid_ratio']

os.makedirs(f"{CACHE}/E-DAIC", exist_ok=True)
os.makedirs(f"{CACHE}/DAIC-WOZ", exist_ok=True)
os.makedirs(f"{CACHE}/D-VLOG", exist_ok=True)

def process_edaic():
    """
    E-DAIC: audio(egemaps25) + face_feat(pose+gaze+AU) + (선택)vgg/densenet
    landmarks는 없음
    """
    index_paths = []
    egemaps = glob.glob(f"{PROC}/E-DAIC/ReEgemaps25LLD/*_egemaps_25lld.csv")
    for eg in egemaps:
        sid = os.path.basename(eg).split('_')[0]
        modal = {"audio": eg}

        # face_feat (pose+gaze+AUs)
        ff = glob.glob(f"{PROC}/E-DAIC/Features/{sid}_openface_pose_gaze_au.csv")
        if ff: modal["face_feat"] = ff[0]

        # (옵션) CNN
        vgg = glob.glob(f"{PROC}/E-DAIC/Features/{sid}_vgg16.csv")
        if vgg: modal["vgg"] = vgg[0]
        den = glob.glob(f"{PROC}/E-DAIC/Features/{sid}_densenet201.csv")
        if den: modal["densenet"] = den[0]

        idx = build_windows(
            session_id=sid,
            modal_csvs=modal,
            win_len=L, stride=S, base_hz=100,
            out_dir=f"{CACHE}/E-DAIC",
            min_audio_ratio=R,
            normalize_landmarks_method=None  # landmarks 자체가 없으므로
        )
        index_paths.append(idx)

    if index_paths:
        df = pd.concat([pd.read_csv(p) for p in index_paths], ignore_index=True)
        df.to_csv(f"{CACHE}/E-DAIC/E-DAIC_all_index.csv", index=False)
        print(f"[3_build_cache] E-DAIC: {len(df)} windows")

def process_daic_woz():
    """
    DAIC-WOZ: audio(egemaps25) + landmarks(CLNF)
    face_feat는 없음
    """
    index_paths = []
    egemaps = glob.glob(f"{PROC}/DAIC-WOZ/ReEgemaps25LLD/*_egemaps_25lld.csv")
    for eg in egemaps:
        sid = os.path.basename(eg).split('_')[0]
        modal = {"audio": eg}

        # CLNF landmarks
        clnf = glob.glob(f"{PROC}/DAIC-WOZ/Features/{sid}_CLNF_features.txt")
        if clnf: modal["landmarks"] = clnf[0]

        idx = build_windows(
            session_id=sid,
            modal_csvs=modal,
            win_len=L, stride=S, base_hz=100,
            out_dir=f"{CACHE}/DAIC-WOZ",
            min_audio_ratio=R,
            normalize_landmarks_method="interocular"
        )
        index_paths.append(idx)

    if index_paths:
        df = pd.concat([pd.read_csv(p) for p in index_paths], ignore_index=True)
        df.to_csv(f"{CACHE}/DAIC-WOZ/DAIC-WOZ_all_index.csv", index=False)
        print(f"[3_build_cache] DAIC-WOZ: {len(df)} windows")

def process_dvlog():
    """
    D-VLOG: 별도 로더/캐시 로직(세션 길이 기반)으로 처리 권장.
    일단은 보류(필요시 acoustic npy를 가상 타임라인으로 래핑해서 windowing).
    """
    print("[3_build_cache] D-VLOG: (보류) 기존 파이프라인 유지")

process_edaic()
process_daic_woz()
process_dvlog()
PY
