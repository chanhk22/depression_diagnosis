# preprocessing/daic_audio_pipeline.py
import os, re, wave, contextlib
import pandas as pd

ELLIE_REGEX_DEFAULT = r"(?i)hi\s*i[' ]?m\s*ellie"

def find_ellie_start(transcript_csv, ellie_regex=ELLIE_REGEX_DEFAULT):
    # DAIC/E-DAIC transcript는 보통 '\t' 구분 + 'value' 컬럼.
    df = pd.read_csv(transcript_csv, delimiter='\t')
    # value 컬럼에서 정규식 매칭
    m = df['value'].astype(str).str.contains(ellie_regex, na=False)
    if not m.any():
        return 0.0
    # 첫 매칭 행의 start_time 사용
    return float(df.loc[m, 'start_time'].iloc[0])

def trim_wav_from_start(in_wav, out_wav, start_sec):
    if start_sec <= 0:
        # 그냥 복사
        import shutil; shutil.copyfile(in_wav, out_wav); return
    with contextlib.closing(wave.open(in_wav, 'rb')) as w:
        fr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        n  = w.getnframes()
        start_frame = int(start_sec * fr)
        start_frame = max(0, min(start_frame, n))
        w.setpos(start_frame)
        frames = w.readframes(n - start_frame)

    with contextlib.closing(wave.open(out_wav, 'wb')) as wout:
        wout.setnchannels(ch)
        wout.setsampwidth(sw)
        wout.setframerate(fr)
        wout.writeframes(frames)

def process_session(audio_wav, transcript_csv, out_wav, ellie_regex=ELLIE_REGEX_DEFAULT):
    t0 = find_ellie_start(transcript_csv, ellie_regex=ellie_regex)
    trim_wav_from_start(audio_wav, out_wav, t0)
    return t0
