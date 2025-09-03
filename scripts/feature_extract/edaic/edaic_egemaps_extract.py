import os
import yaml
import json
import glob
from features.batch_extract_egemaps import extract_egemaps_lld  # batch_extract 안쓰고 직접 호출

# 설정 불러오기
with open("configs/default.yaml", encoding="utf-8") as f: 
    C = yaml.safe_load(f)

AUD_IN = C['paths']['e_daic']['audio_dir']
PROC   = C['outputs']['processed_root']

# t0.json 로드
with open("t0_values.json", "r") as f:
    t0_values = json.load(f)

exclude_sids = set(t0_values.keys())

# 입력/출력 디렉토리
out_dir   = os.path.join(PROC, "E-DAIC", "ReEgemaps25LLD")
os.makedirs(out_dir, exist_ok=True)

# 오디오 파일 순회
all_audio_files = glob.glob(os.path.join(AUD_IN, "*_AUDIO.wav"))
processed = 0

for wav_file in all_audio_files:
    fname = os.path.basename(wav_file)
    sid = fname.split("_")[0]  # "490_AUDIO.wav" -> "490"

    # t0.json에 있으면 건너뜀
    if sid in exclude_sids:
        continue

    out_csv = os.path.join(out_dir, f"{sid}_egemaps_25lld.csv")

    if os.path.exists(out_csv):
        continue

    try:
        extract_egemaps_lld(wav_file, out_csv)
        processed += 1
    except Exception as e:
        print(f"Error extracting eGeMAPS for {sid}: {e}")

print(f"eGeMAPS 25 LLD re-extracted for {processed} audio files (excluding t0.json sids).")
