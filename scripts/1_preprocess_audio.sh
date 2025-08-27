#!/usr/bin/env bash
set -e
CONF= configs/default.yaml

# 예: 파이썬 스크립트로 세션 루프 돌며
python - << 'PY'
import os, yaml, glob
from preprocessing.daic_audio_pipeline import find_ellie_start, trim_wav_from_start, process_session

with open("configs/default.yaml") as f: C=yaml.safe_load(f)
AUDIO_IN  = C['paths']['daic_woz']['audio_dir']
TRS_IN    = C['paths']['daic_woz']['transcript_dir']
FEAT_IN   = C['paths']['daic_woz']['features_dir']
PROC_ROOT = C['outputs']['processed_root']
ELLIE_RGX = C['preprocessing']['ellie_regex']

os.makedirs(f"{PROC_ROOT}/DAIC-WOZ/Audio", exist_ok=True)
os.makedirs(f"{PROC_ROOT}/DAIC-WOZ/Features", exist_ok=True)

for trs in glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv")):
    sid = os.path.basename(trs).split('_')[0]
    wav_in  = os.path.join(AUDIO_IN,  f"{sid}_AUDIO.wav")
    wav_out = os.path.join(PROC_ROOT, "DAIC-WOZ/Audio", f"{sid}_AUDIO_trimmed.wav")
    if not os.path.exists(wav_in): continue
    t0 = find_ellie_start(trs, ELLIE_RGX)
    trim_wav_from_start(wav_in, wav_out, t0)

    # DAIC per-frame CSV들도 동일하게 t>=t0 필터 (예: 기존 egemaps/mfcc/openface가 있다면)
    for csv_in in glob.glob(os.path.join(FEAT_IN, f"{sid}_*.csv")):
        out_csv = os.path.join(PROC_ROOT, "DAIC-WOZ/Features", os.path.basename(csv_in))
        process_session(csv_in, out_csv, t0)
print("DAIC head cut + CSV t0 filter done.")
PY
