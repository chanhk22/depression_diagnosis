#!/usr/bin/env bash
set -e

python - <<'PY'
import os, glob, yaml
from features.batch_extract_egemaps import batch_extract

with open("configs/default.yaml") as f: C=yaml.safe_load(f)
PROC = C['outputs']['processed_root']

# DAIC-WOZ: t0 컷된 오디오 기준 25 LLD 재추출
batch_extract(f"{PROC}/DAIC-WOZ/Audio", f"{PROC}/DAIC-WOZ/Features", pattern="*_AUDIO_trimmed.wav")

# E-DAIC: 컷 불필요하나 통일성 위해 재추출(원한다면 Audio 원본 사용)
batch_extract(f"{C['paths']['e_daic']['audio_dir']}", f"{PROC}/E-DAIC/Features", pattern="*_AUDIO.wav")
print("eGeMAPS 25 LLD re-extracted.")
PY
