import os, pandas as pd
from .base_dataset import BaseDataset

class DVLOGDataset(BaseDataset):
    def __init__(self, split="train", config=None, label_type="binary"):
        df = pd.read_csv(config["labels"]["dvlog"]["labels_csv"])
        df = df[df["fold"] == split]

        idx_dir = f"{config['outputs']['cache_root']}/D-VLOG"
        index_csvs = [os.path.join(idx_dir, f) for f in os.listdir(idx_dir) if f.endswith("index.csv")]
        all_idx = pd.concat([pd.read_csv(f) for f in index_csvs])

        merged = all_idx.merge(df, left_on="session", right_on="index", how="inner")
        merged["y_bin"] = (merged["label"] == "depression").astype(int)
        merged["y_reg"] = 0.0   # DVLOG에는 점수 없음

        index_csv = f"{idx_dir}/{split}_index.csv"
        merged.to_csv(index_csv, index=False)

        super().__init__(index_csv=index_csv, label_type=label_type)
