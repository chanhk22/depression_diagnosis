import yaml
from features.batch_extract_egemaps import batch_extract

with open("configs/default.yaml", encoding='utf-8') as f: 
    C=yaml.safe_load(f)
PROC = C['outputs']['processed_root']

# DAIC-WOZ: t0 컷된 오디오 기준 25 LLD 재추출
batch_extract(f"{PROC}/DAIC-WOZ/Audio", f"{PROC}/DAIC-WOZ/ReEgemaps25LLD", pattern="*_AUDIO_trimmed.wav")


print("eGeMAPS 25 LLD re-extracted.")