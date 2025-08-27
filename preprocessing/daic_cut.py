# preprocessing/daic_cut_apply.py
from .daic_audio_pipeline import process_session
from .mask_csv import mask_by_t0

def run_daic_cut_one(session_id, wav_in, trans_csv, out_wav, csv_list_in, csv_list_out):
    t0 = process_session(wav_in, trans_csv, out_wav)  # 오디오 앞부분 컷
    for inp, outp in zip(csv_list_in, csv_list_out):
        mask_by_t0(inp, outp, t0)                    # per-frame CSV t ≥ t0만
    return t0
