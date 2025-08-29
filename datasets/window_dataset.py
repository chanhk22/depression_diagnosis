import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    index_csv produced by build_windows contains 'path' column pointing to per-window .npz
    Each .npz may contain keys: audio, landmarks, mfcc, vgg, dense, ...
    """
    def __init__(self, index_csv, modal_keys=("audio","landmarks","mfcc","vgg","dense")):
        self.df = pd.read_csv(index_csv)
        self.modal_keys = modal_keys

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = r["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        npz = np.load(p, allow_pickle=True)
        sample = {}
        for k in self.modal_keys:
            if k in npz:
                val = npz[k]
                # landmarks: if (win,68,2), flatten to (win,136) for model convenience
                if k == "landmarks" and val is not None:
                    arr = np.asarray(val)
                    if arr.ndim == 3 and arr.shape[1]==68 and arr.shape[2]==2:
                        arr = arr.reshape(arr.shape[0], -1)
                    sample[k] = arr.astype(np.float32)
                else:
                    sample[k] = np.asarray(val).astype(np.float32)
            else:
                sample[k] = None

        sample['y_bin'] = float(r["y_bin"]) if "y_bin" in r and not pd.isna(r["y_bin"]) else None
        sample['y_reg'] = float(r["y_reg"]) if "y_reg" in r and not pd.isna(r["y_reg"]) else None
        sample['meta'] = {"session": r.get("session"), "t0": r.get("t0"), "t1": r.get("t1")}
        return sample
