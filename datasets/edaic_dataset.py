import os, pandas as pd
from .base_dataset import BaseDataset

class EDAICDataset(BaseDataset):
    def __init__(self, split="train", config=None, label_type="binary"):
        label_path = config["labels"]["e_daic"][f"{split}_split"]
        df = pd.read_csv(label_path)

        idx_dir = f"{config['outputs']['cache_root']}/E-DAIC"
        index_csvs = [os.path.join(idx_dir, f) for f in os.listdir(idx_dir) if f.endswith("index.csv")]
        all_idx = pd.concat([pd.read_csv(f) for f in index_csvs])

        merged = all_idx.merge(df, left_on="session", right_on="Participant_ID", how="inner")
        merged["y_bin"] = merged["PHQ_Binary"]
        merged["y_reg"] = merged["PHQ_Score"]

        index_csv = f"{idx_dir}/{split}_index.csv"
        merged.to_csv(index_csv, index=False)

        super().__init__(index_csv=index_csv, label_type=label_type)
