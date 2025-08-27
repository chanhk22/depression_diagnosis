# datasets/window_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np, pandas as pd

class WindowDataset(Dataset):
    def __init__(self, index_csv, transform=None):
        self.df = pd.read_csv(index_csv)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'], allow_pickle=True)
        # audio -> (T,25)
        audio = data['audio'].astype(np.float32) if 'audio' in data else np.zeros((1,25),np.float32)
        visual = data['visual'].astype(np.float32) if 'visual' in data else np.zeros((1,136),np.float32)
        # Labels/Meta: derive from session-based label tables (user should implement mapping)
        sample = {"audio":torch.from_numpy(audio), "visual":torch.from_numpy(visual), "meta":row.to_dict()}
        if self.transform: sample = self.transform(sample)
        return sample
