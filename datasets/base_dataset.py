import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class BaseDataset(Dataset):
    def __init__(self, index_csv, label_type="binary"):
        self.df = pd.read_csv(index_csv)
        self.label_type = label_type

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = dict(np.load(row["path"], allow_pickle=True))

        # label
        if self.label_type == "binary":
            y = float(row["y_bin"])
        elif self.label_type == "regression":
            y = float(row["y_reg"])
        else:
            raise ValueError(f"Unknown label_type {self.label_type}")

        meta = {
            "session": row.get("session", ""),
            "dataset": row.get("dataset", ""),
            "gender":  row.get("gender", ""),
        }
        return data, y, meta
