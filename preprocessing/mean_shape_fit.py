import numpy as np, glob, os, yaml
from pathlib import Path
import argparse

def procrustes_align(X, Y):
    # X: (N,2) source, Y: (N,2) target mean shape
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt
    s = np.trace((Xc @ R).T @ Yc) / np.trace(Xc.T @ Xc)
    t = Y.mean(0) - s*(X.mean(0)@R)
    return s, R, t

def apply_align(seq, mean_shape):
    # seq: (T,68,2)
    T = seq.shape[0]; out = np.empty_like(seq)
    for t in range(T):
        s,R,tv = procrustes_align(seq[t], mean_shape)
        out[t] = (seq[t]@R)*s + tv
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args(); env = yaml.safe_load(open(args.env))
    lm_dir = Path(env["outputs"]["lmk_dir"])
    # 학습 split의 DAIC + D-VLOG 랜드마크를 샘플링해서 mean shape 산출
    files = glob.glob(os.path.join(lm_dir, "daic__*.npy")) + glob.glob(os.path.join(lm_dir, "dvlog__*.npy"))
    sample = []
    for f in files[:min(500, len(files))]:
        arr = np.load(f)
        idx = np.random.randint(0, arr.shape[0])
        sample.append(arr[idx])
    mean_shape = np.mean(np.stack(sample,0), axis=0)  # (68,2)
    np.save(lm_dir / "mean_shape_68.npy", mean_shape)


import numpy as np, yaml, glob, os
from pathlib import Path

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args(); env = yaml.safe_load(open(args.env))
    lm_dir = Path(env["paths"]["cache_root"]) / env["outputs"]["lmk_dir"]
    files = glob.glob(str(lm_dir / "daic__*.npy")) + glob.glob(str(lm_dir / "dvlog__*.npy"))
    sample = []
    for f in files[:min(1000, len(files))]:
        arr = np.load(f)
        idx = np.random.randint(0, arr.shape[0])
        sample.append(arr[idx])
    mean_shape = np.mean(np.stack(sample,0), axis=0)  # (68,2)
    out = Path(env["paths"]["cache_root"]) / env["outputs"]["mean_shape"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mean_shape.astype(np.float32))
