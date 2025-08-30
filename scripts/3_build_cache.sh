#!/usr/bin/env bash
set -e

python - <<'PY'
import os, yaml, glob, pandas as pd
from precache.window_cache import build_windows

with open("configs/default.yaml") as f:
    C = yaml.safe_load(f)

PROC   = C['outputs']['processed_root']
CACHE  = C['outputs']['cache_root']
L      = C['windowing']['length_s']
S      = C['windowing']['stride_s']
R      = C['windowing']['min_valid_ratio']
BASEHZ = C['windowing']['base_rate_hz']
LMNORM = C['preprocessing'].get('normalize_landmarks', None)

os.makedirs(f"{CACHE}/E-DAIC", exist_ok=True)
os.makedirs(f"{CACHE}/DAIC-WOZ", exist_ok=True)

def process_dataset(name, egemaps_glob, cache_dir, extra_patterns=None):
    index_paths = []
    for eg in glob.glob(egemaps_glob):
        sid = os.path.basename(eg).split('_')[0]
        modal = {"audio": eg}

        # 선택: 존재하는 특권 모달 부착
        if extra_patterns:
            for k, pat in extra_patterns.items():
                cand = glob.glob(pat.replace("{sid}", sid))
                if cand:
                    modal[k] = cand[0]

        idx_path = build_windows(
            session_id=sid,
            modal_csvs=modal,               # dict: modality -> path
            win_len=L,
            stride=S,
            base_hz=BASEHZ,
            out_dir=cache_dir,
            min_audio_ratio=R,
            normalize_landmarks_method=LMNORM
        )
        index_paths.append(idx_path)

    if index_paths:
        df = pd.concat([pd.read_csv(p) for p in index_paths], ignore_index=True)
        out_csv = os.path.join(cache_dir, f"{name}_all_index.csv")
        df.to_csv(out_csv, index=False)
        print(f"[3_build_cache] {name}: {len(df)} windows -> {out_csv}")
    else:
        print(f"[3_build_cache] {name}: no sessions found for {egemaps_glob}")

# E-DAIC (재추출 25 LLD 기준)
process_dataset(
    "E-DAIC",
    egemaps_glob=f"{PROC}/E-DAIC/ReEgemaps25LLD/*_egemaps_25lld.csv",
    cache_dir=f"{CACHE}/E-DAIC",
    extra_patterns={
        # OpenFace (pose/gaze/AUs) — 컬럼 모두 숫자면 일반 timeseries로 자동 취급
        "openface":  f"{PROC}/E-DAIC/Features/{{sid}}_openface_pose_gaze_AUs.csv",
        # CNN 1Hz 특징
        "vgg16":     f"{PROC}/E-DAIC/Features/{{sid}}_vgg16.csv",
        "densenet":  f"{PROC}/E-DAIC/Features/{{sid}}_densenet201.csv",
        # MFCC(특권)
        "mfcc":      f"{PROC}/E-DAIC/Features/{{sid}}_OpenSMILE2.3.0_mfcc.csv",
    }
)

# DAIC-WOZ (재추출 25 LLD + CLNF/COVAREP)
process_dataset(
    "DAIC-WOZ",
    egemaps_glob=f"{PROC}/DAIC-WOZ/ReEgemaps25LLD/*_egemaps_25lld.csv",
    cache_dir=f"{CACHE}/DAIC-WOZ",
    extra_patterns={
        # CLNF landmarks txt는 별도 파서로 CSV화 했다면 그 경로를 연결
        # (ex) {sid}_CLNF_features_clean.csv 로 전처리 저장했다고 가정
        "landmarks": f"{PROC}/DAIC-WOZ/Features/{{sid}}_CLNF_features_clean.csv",
        "covarep":   f"{PROC}/DAIC-WOZ/Features/{{sid}}_COVAREP.csv",
    }
)

print("[3_build_cache] Done. (D-VLOG는 3b 스크립트 사용)")
PY
