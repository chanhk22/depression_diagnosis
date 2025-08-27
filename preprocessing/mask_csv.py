# preprocessing/mask_csv_by_transcript.py
import os
from .utils_io import read_table_smart

def mask_by_t0(in_csv, out_csv, t0):
    df, sep = read_table_smart(in_csv)
    # 시간 컬럼 후보
    time_cols = [c for c in ['frameTime','timestamp','timeStamp','time','FrameTime','frame_time'] if c in df.columns]
    if not time_cols:
        # 1Hz 특징(vgg/densenet)은 timeStamp가 보통 있음
        # 없으면 그대로 복사
        df.to_csv(out_csv, index=False)
        return
    tcol = time_cols[0]
    # 정수형 timeStamp(초)면 float로 변환
    if str(df[tcol].dtype).startswith('int'):
        df[tcol] = df[tcol].astype(float)
    # t0 이후만
    df = df[df[tcol] >= float(t0)]
    df.to_csv(out_csv, index=False)
