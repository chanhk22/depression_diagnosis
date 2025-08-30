import os, pandas as pd
from .base_dataset import BaseDataset

class DAICDataset(BaseDataset):
    def __init__(self, split="train", config=None, label_type="binary"):
        label_path = config["labels"]["daic_woz"][f"{split}_split"]
        df = pd.read_csv(label_path)

        # 여기서 index.csv 매칭 (cache 단계에서 만든 파일들)
        index_csvs = [f for f in os.listdir(f"{config['outputs']['cache_root']}/DAIC-WOZ") if f.endswith("index.csv")]
        all_idx = pd.concat([pd.read_csv(os.path.join(config['outputs']['cache_root'], "DAIC-WOZ", f)) for f in index_csvs])
        
        merged = all_idx.merge(df, left_on="session", right_on="Participant_ID", how="inner")
        merged["y_bin"] = merged["PHQ8_Binary"]
        merged["y_reg"] = merged["PHQ8_Score"]

        index_csv = f"{config['outputs']['cache_root']}/DAIC-WOZ/{split}_index.csv"
        merged.to_csv(index_csv, index=False)

        super().__init__(index_csv=index_csv, label_type=label_type)
