#lld -> 윈도 통계
# features/audio/egemaps_reader.py
import numpy as np
import pandas as pd

EGEMAPS_25 = [
  # 열 이름을 데이터에 맞춰 정렬(EDAIC/opensmile csv의 LLD 25개만)
  # 예시: 'F0semitoneFrom27.5Hz_sma3nz', ...
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3"
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3amplitudeLogRelF0_sma3nz"
]

def read_ege_llD_from_csv(path_csv) -> np.ndarray:
    df = pd.read_csv(path_csv)
    # E-DAIC의 eGeMAPS CSV는 프레임별 row, 다양한 컬럼 포함.
    # 25 LLD만 선택:
    cols = [c for c in df.columns if c in EGEMAPS_25]
    x = df[cols].values.astype(np.float32)
    x[np.isnan(x)] = 0.0
    return x  # (T, 25)
