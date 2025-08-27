#!/usr/bin/env bash
set -e

python - <<'PY'
import os, yaml, glob
from precache.window_cache import build_windows

with open("configs/default.yaml") as f: C=yaml.safe_load(f)
PROC = C['outputs']['processed_root']
CACHE= C['outputs']['cache_root']
L,S  = C['windowing']['length_s'], C['windowing']['stride_s']
R    = C['windowing']['min_audio_ratio']

os.makedirs(f"{CACHE}/E-DAIC", exist_ok=True)
os.makedirs(f"{CACHE}/DAIC-WOZ", exist_ok=True)

# E-DAIC: audio only(25 LLD) + (선택) 특권 CSV들도 modal로 넣을 수 있음
for eg in glob.glob(f"{PROC}/E-DAIC/Features/*_OpenSMILE2.3.0_egemaps_25lld.csv"):
    sid = os.path.basename(eg).split('_')[0]
    modal = {"audio": eg}
    # 필요 시: openface/au/pose/gaze/vgg/dense 추가
    build_windows
PY