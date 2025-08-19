import numpy as np

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))

def eye_aspect_ratio(pts):
    # pts: (68,2)
    def dist(a,b): return np.linalg.norm(pts[a]-pts[b])
    L = LEFT_EYE; R = RIGHT_EYE
    earL = (dist(L[1],L[5])+dist(L[2],L[4]))/(2.0*dist(L[0],L[3]) + 1e-8)
    earR = (dist(R[1],R[5])+dist(R[2],R[4]))/(2.0*dist(R[0],R[3]) + 1e-8)
    return (earL+earR)/2.0

def mouth_aspect_ratio(pts):
    M = MOUTH
    def dist(a,b): return np.linalg.norm(pts[a]-pts[b])
    return (dist(61,67)+dist(62,66)+dist(63,65))/(3.0*dist(60,64) + 1e-8)

def normalize_frame(pts):
    # 눈 중심 정렬 + 눈간거리 스케일 + yaw/pitch 간단 회전보정(여기선 2D 회전만)
    eye_center = (pts[36:42].mean(0) + pts[42:48].mean(0))/2.0
    interocular = np.linalg.norm(pts[36]-pts[45]) + 1e-8
    p = (pts - eye_center) / interocular
    return p

def extract_micro(seq, fps=25.0):
    # seq: (T,68,2) normalized
    T = seq.shape[0]
    EAR = np.array([eye_aspect_ratio(f) for f in seq], dtype=np.float32)
    MAR = np.array([mouth_aspect_ratio(f) for f in seq], dtype=np.float32)
    # head velocity: 코 끝(30) 프레임 차분 크기
    nose = seq[:,30,:]
    vel = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    head_vel = np.concatenate([[0.0], vel])*fps
    # blink: EAR < thresh (적당히 백분위수 기반)
    thr = np.percentile(EAR, 20)
    blink = (EAR < thr).astype(np.float32)
    return {"EAR": EAR, "MAR": MAR, "head_vel": head_vel, "blink": blink}
