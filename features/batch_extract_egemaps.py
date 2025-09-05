import os, glob
from .egemaps_extract import extract_egemaps_lld

def batch_extract(in_wav_dir, out_dir, pattern="*_AUDIO_trimmed.wav"):
    os.makedirs(out_dir, exist_ok=True)
    for w in glob.glob(os.path.join(in_wav_dir, pattern)):
        sid = os.path.basename(w).split('_')[0]
        out_csv = os.path.join(out_dir, f"{sid}_egemaps_25lld.csv")
        # 이미 해당 파일이 존재하면 건너뛰기
        if os.path.basename(out_csv) in set(os.listdir(out_dir)):
            print(f"Skipping already processed file: {out_csv}")
            continue  # 이미 처리된 파일은 건너뜀
        extract_egemaps_lld(w, out_csv)

