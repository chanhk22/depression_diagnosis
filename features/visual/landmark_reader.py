# features/visual/landmark_reader.py
import numpy as np
import re

def parse_clnf_features_txt(path_txt) -> np.ndarray:
    """
    DAIC-WOZ CLNF features.txt에서 2D 68x2 좌표만 추출하여 (T, 136) 반환.
    파일 포맷은 OpenFace CLNF 출력을 가정. 라인별로 x_0..x_67,y_0..y_67 형태.
    """
    xs = []
    with open(path_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): 
                continue
            vals = re.split(r'[,\s]+', line.strip())
            vals = [v for v in vals if v]
            # 포맷 차이가 있으면 여기서 인덱스를 조정
            arr = np.array(list(map(float, vals)), dtype=np.float32)
            # 필요 컬럼 슬라이싱: 예) [x0..x67, y0..y67] -> 136
            # TODO: 실제 컬럼 위치에 맞게 인덱싱
            xs.append(arr[:136])
    x = np.stack(xs, 0) if xs else np.zeros((0, 136), np.float32)
    x[np.isnan(x)] = 0.0
    return x

def read_dvlog_npy(path_npy) -> np.ndarray:
    x = np.load(path_npy).astype(np.float32)
    x[np.isnan(x)] = 0.0
    return x  # (T, 136)
