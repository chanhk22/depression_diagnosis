import numpy as np, pandas as pd, yaml, glob, os
from pathlib import Path
import opensmile

def extract_lld(wav_path: Path) -> np.ndarray:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    df = smile.process_file(str(wav_path))   # (T,25)
    return df.values.astype(np.float32)

def save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args()
    env = yaml.safe_load(open(args.env))

    out_root = Path(env["paths"]["cache_root"]) / env["outputs"]["audio_lld_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    # E-DAIC
    ed_root = Path(env["paths"]["edaic_root"])
    ed_wavs = glob.glob(os.path.join(ed_root, env["edaic"]["wav_glob"]), recursive=True)
    for w in ed_wavs:
        arr = extract_lld(Path(w))
        save_npy(out_root / f"edaic__{Path(w).stem}.npy", arr)

    # DAIC
    da_root = Path(env["paths"]["daic_root"])
    da_wavs = glob.glob(os.path.join(da_root, env["daic"]["wav_glob"]), recursive=True)
    for w in da_wavs:
        arr = extract_lld(Path(w))
        save_npy(out_root / f"daic__{Path(w).stem}.npy", arr)
