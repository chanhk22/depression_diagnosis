import numpy as np

def normalize_visual_npy(path_npy):
    v = np.load(path_npy)  # (T, 68*2) or (T, 136)
    if v.ndim==2 and v.shape[1]==136:
        pts = v.reshape(-1,68,2).astype(np.float32)
    else:
        raise ValueError("Unexpected shape for visual.npy")
    # 같은 규칙으로 표준화
    c = pts.mean(axis=1, keepdims=True)
    p = pts - c
    L=[36,37,38,39,40,41]; R=[42,43,44,45,46,47]
    le=p[:,L,:].mean(axis=1); re=p[:,R,:].mean(axis=1)
    scale=np.linalg.norm(re-le,axis=1,keepdims=True)+1e-6
    p = p/scale[:,None,None]
    return p  # (T,68,2)
