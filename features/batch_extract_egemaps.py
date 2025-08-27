import os, glob
from .egemaps_extract import extract_egemaps_lld

def batch_extract(in_wav_dir, out_dir, pattern="*_AUDIO_trimmed.wav"):
    os.makedirs(out_dir, exist_ok=True)
    for w in glob.glob(os.path.join(in_wav_dir, pattern)):
        sid = os.path.basename(w).split('_')[0]
        out_csv = os.path.join(out_dir, f"{sid}_OpenSMILE2.3.0_egemaps_25lld.csv")
        extract_egemaps_lld(w, out_csv)
