# datasets/dvlog.py
import os, numpy as np, pandas as pd
from torch.utils.data import Dataset
from features.visual.landmark_reader import read_dvlog_npy

class DVLOGDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        meta_csv = cfg["data"]["datasets"]["dvlog"]["meta_csv"]
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["fold"]==split]
        self.npy_root = cfg["data"]["datasets"]["dvlog"]["npy_root"]

    def __len__(self): return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        pid = str(row["id"])
        x_a = np.load(os.path.join(self.npy_root, pid, "acoustic.npy")).astype(np.float32)  # (T,25)
        x_v = read_dvlog_npy(os.path.join(self.npy_root, pid, "visual.npy"))               # (T,136)
        y = float(row["label"])  # 0/1
        return {"audio": x_a, "landmark": x_v, "label": y, "meta": {"id": pid, "src": "D-VLOG"}}
