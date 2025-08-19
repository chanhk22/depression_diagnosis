import numpy as np, pandas as pd, yaml
from pathlib import Path

def save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args()
    env = yaml.safe_load(open(args.env))

    root = Path(env["paths"]["dvlog_root"])
    meta = pd.read_csv(root / env["dvlog"]["meta_csv"])
    out_audio = Path(env["paths"]["cache_root"]) / env["outputs"]["audio_lld_dir"]
    out_lmk   = Path(env["paths"]["cache_root"]) / env["outputs"]["lmk_dir"]
    out_audio.mkdir(parents=True, exist_ok=True); out_lmk.mkdir(parents=True, exist_ok=True)

    for r in meta.itertuples():
        a = np.load(root / env["dvlog"]["acoustic_dir"] / f"{r.id}.npy")  # (T,25)
        v = np.load(root / env["dvlog"]["visual_dir"] / f"{r.id}.npy")    # (T,68,2)
        save(out_audio / f"dvlog__{r.id}.npy", a)
        save(out_lmk   / f"dvlog__{r.id}.npy", v)
