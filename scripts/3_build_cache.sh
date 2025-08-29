#!/usr/bin/env bash
set -e

python - <<'PY'
import os, yaml, glob, pandas as pd
from precache.window_cache import build_windows

with open("configs/default.yaml") as f:
    C = yaml.safe_load(f)

PROC  = C['paths']['processed_root']
CACHE = C['paths']['cache_root']
L     = C['windowing']['length_s']
S     = C['windowing']['stride_s']
R     = C['windowing']['min_audio_ratio']

os.makedirs(f"{CACHE}/E-DAIC", exist_ok=True)
os.makedirs(f"{CACHE}/DAIC-WOZ", exist_ok=True)
os.makedirs(f"{CACHE}/D-VLOG", exist_ok=True)

def process_dataset(dataset_name, pattern, cache_subdir):
    """
    dataset_name: str ("E-DAIC", "DAIC-WOZ", "D-VLOG")
    pattern: glob pattern for egemaps csv
    cache_subdir: path under CACHE
    """
    index_paths = []
    for eg in glob.glob(pattern):
        sid = os.path.basename(eg).split('_')[0]
        modal = {"audio": eg}
        # TODO: 여기에 openface / vgg / densenet / pose 등 필요한 modality 경로를 추가해도 됨
        idx_path = build_windows(
            session_id=sid,
            modal_csvs=modal,
            win_len=L,
            stride=S,
            base_hz=100,
            out_dir=cache_subdir,
            min_audio_ratio=R
        )
        index_paths.append(idx_path)
    # merge all index.csv
    if index_paths:
        dfs = [pd.read_csv(p) for p in index_paths]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(cache_subdir, f"{dataset_name}_all_index.csv"), index=False)
        print(f"[3_build_cache] {dataset_name}: built {len(df_all)} windows")

# === Run for each dataset ===
process_dataset("E-DAIC", f"{PROC}/E-DAIC/Features/*_OpenSMILE2.3.0_egemaps_25lld.csv", f"{CACHE}/E-DAIC")
process_dataset("DAIC-WOZ", f"{PROC}/DAIC-WOZ/Features/*_OpenSMILE2.3.0_egemaps_25lld.csv", f"{CACHE}/DAIC-WOZ")
process_dataset("D-VLOG", f"{PROC}/D-VLOG/Features/*_OpenSMILE2.3.0_egemaps_25lld.csv", f"{CACHE}/D-VLOG")
PY
