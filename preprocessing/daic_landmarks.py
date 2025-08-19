import numpy as np, pandas as pd, yaml, glob, os
from pathlib import Path
from .landmarks_utils import normalize_frame, extract_micro

def load_clnf_txt(p: Path) -> np.ndarray:
    # OpenFace CLNF_features.txt: 첫 줄 메타, 이후 x_*, y_* 칼럼
    df = pd.read_csv(p, sep=",", skiprows=1)
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    x = df[x_cols].values
    y = df[y_cols].values
    pts = np.stack([x, y], axis=2)  # (T,68,2)
    return pts.astype(np.float32)

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args(); env = yaml.safe_load(open(args.env))

    da_root = Path(env["paths"]["daic_root"])
    out_lmk = Path(env["paths"]["cache_root"]) / env["outputs"]["lmk_dir"]
    out_micro = Path(env["paths"]["cache_root"]) / env["outputs"]["micro_dir"]
    out_lmk.mkdir(parents=True, exist_ok=True); out_micro.mkdir(parents=True, exist_ok=True)

    lmks = glob.glob(os.path.join(da_root, env["daic"]["clnf_lmk_glob"]), recursive=True)
    for p in lmks:
        seq = load_clnf_txt(Path(p))
        norm_seq = np.stack([normalize_frame(f) for f in seq], axis=0)
        feats = extract_micro(norm_seq, fps=25.0)
        np.save(out_lmk / f"daic__{Path(p).stem}.npy", norm_seq.astype(np.float32))
        np.savez_compressed(out_micro / f"daic__{Path(p).stem}.npz", **{k:v.astype(np.float32) for k,v in feats.items()})
