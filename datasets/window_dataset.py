# datasets/window_dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    index_csv: rows with columns ['file', 'domain', 'pid', 'split', 'label']
    Each 'file' is a .npz saved window (lld, optional lmk, micro, priv, label, domain, pid)
    """
    def __init__(self, index_csv, split="train", domains=None):
        self.df = pd.read_csv(index_csv)
        self.df = self.df[self.df['split'] == split]
        if domains is not None:
            self.df = self.df[self.df['domain'].isin(domains)]
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = dict(np.load(row['file'], allow_pickle=True))
        # ensure keys exist: label/domain/pid/ etc
        data['domain'] = row['domain']
        data['pid'] = row['pid']
        data['split'] = row['split']
        data['file'] = row['file']
        # label may be present in file or in csv, ensure integer
        data['label'] = int(row['label'])
        return data
