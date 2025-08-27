# preprocessing/clnf_parser.py
import numpy as np
import pandas as pd
from .utils_io import read_table_smart

IBUG_LEFT_EYE  = [36,37,38,39,40,41]
IBUG_RIGHT_EYE = [42,43,44,45,46,47]
IBUG_MOUTH = [60,61,62,63,64,65,66,67]

def read_clnf_features_txt(path):
    # header: frame, timestamp, confidence, success, x0..x67, y0..y67
    df, _ = read_table_smart(path)
    # 열 이름 정리
    cols = df.columns.tolist()
    # x0가 어디서 시작하는지 찾기
    x0_idx = next(i for i,c in enumerate(cols) if c.startswith('x0') or c=='x_0' or c=='x0')
    x = df.iloc[:, x0_idx:x0_idx+68].to_numpy(dtype=np.float32)
    y = df.iloc[:, x0_idx+68:x0_idx+136].to_numpy(dtype=np.float32)
    t = df['timestamp'].to_numpy(dtype=np.float32) if 'timestamp' in df.columns else None
    pts = np.stack([x,y], axis=-1)  # (T,68,2)
    return t, pts  # 픽셀 좌표

def normalize_landmarks(pts):
    # pts: (T,68,2) in pixel or arbitrary coords → canonical
    c = pts.mean(axis=1, keepdims=True)          # 중심
    p = pts - c
    le = p[:, IBUG_LEFT_EYE, :].mean(axis=1)     # (T,2)
    re = p[:, IBUG_RIGHT_EYE, :].mean(axis=1)
    eye_dist = np.linalg.norm(re - le, axis=1, keepdims=True) + 1e-6
    p = p / eye_dist[:,None,None]
    return p  # (T,68,2) centered & scaled

def ear_mar(pts):
    def eye_ratio(p, idx):
        v1=np.linalg.norm(p[:,idx[1]]-p[:,idx[5]],axis=1)
        v2=np.linalg.norm(p[:,idx[2]]-p[:,idx[4]],axis=1)
        h =np.linalg.norm(p[:,idx[0]]-p[:,idx[3]],axis=1)+1e-6
        return (v1+v2)/(2*h)
    EAR=(eye_ratio(pts,IBUG_LEFT_EYE)+eye_ratio(pts, IBUG_RIGHT_EYE))/2
    p=pts[:,IBUG_MOUTH,:]
    MAR=np.linalg.norm(p[:,2]-p[:,6],axis=1)/(np.linalg.norm(p[:,0]-p[:,4],axis=1)+1e-6)
    return EAR, MAR