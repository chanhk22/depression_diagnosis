# features/audio/egemaps_extract.py
import os
import pandas as pd
import opensmile

def extract_egemaps_lld(wav_path, out_csv):
    # 설치: pip install opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    df = smile.process_file(wav_path).reset_index()  # columns: ['file', 'start', <features>]
    # 표준 컬럼명 맞춤
    df = df.rename(columns={'start':'frameTime', 'file':'name'})
    # name은 파일명으로
    df['name'] = os.path.basename(wav_path)
    # opensmile는 10ms 스텝 → frameTime 그대로 사용
    df.to_csv(out_csv, index=False)
