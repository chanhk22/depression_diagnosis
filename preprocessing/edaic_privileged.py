import numpy as np, pandas as pd, yaml, glob, os
from pathlib import Path

def numeric_df(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].values.astype(np.float32)

def concat_daic_priv(au_csv, pose_txt, gaze_txt):
    au = pd.read_csv(au_csv)
    pose = pd.read_csv(pose_txt, sep=",")
    gaze = pd.read_csv(gaze_txt, sep=",")
    def num(df): 
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return df[cols].values
    arr = np.concatenate([num(au), num(pose), num(gaze)], axis=1).astype(np.float32)
    return arr

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    args = ap.parse_args(); env = yaml.safe_load(open(args.env))
    out = Path(env["paths"]["cache_root"]) / env["outputs"]["priv_dir"]
    out.mkdir(parents=True, exist_ok=True)

    # E-DAIC
    ed_root = Path(env["paths"]["edaic_root"])
    of_csvs = glob.glob(os.path.join(ed_root, env["edaic"]["of_csv_glob"]), recursive=True)
    for c in of_csvs:
        np.save(out / f"edaic__{Path(c).stem}.npy", numeric_df(c))

    # DAIC
    da_root = Path(env["paths"]["daic_root"])
    au_csvs  = glob.glob(os.path.join(da_root, env["daic"]["clnf_au_glob"]), recursive=True)
    for au in au_csvs:
        stem = Path(au).stem.replace("_CLNF_AUs","")
        pose = str(Path(au).with_name(stem + "_CLNF_pose.txt"))
        gaze = str(Path(au).with_name(stem + "_CLNF_gaze.txt"))
        arr = concat_daic_priv(au, pose, gaze)
        np.save(out / f"daic__{stem}.npy", arr)
