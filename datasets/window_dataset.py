#window_dataset.py
import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    index.csv columns:
      path_audio, path_vis, path_priv_vgg, path_priv_densenet, path_priv_aus,
      y_bin, y_reg, gender, session, dataset
    """
    def __init__(self, index_csv):
        self.df = pd.read_csv(index_csv)

    def __len__(self): 
        return len(self.df)

    def _load_or_none(self, p):
        return np.load(p, allow_pickle=True)["x"] if (isinstance(p, str) and os.path.exists(p)) else None

    def __getitem__(self, i):
        r = self.df.iloc[i]
        data = np.load(r["path"], allow_pickle=True)

        audio = data.get("audio")
        landmarks = data.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.reshape(landmarks.shape[0], -1)  # (win,136)
        
        priv = {
            "vgg":      self._load_or_none(r.get("path_priv_vgg")),
            "densenet": self._load_or_none(r.get("path_priv_densenet")),
            "aus":      self._load_or_none(r.get("path_priv_aus")),
        }
        return {
            "audio": audio,
            "vis": vis,
            "priv": priv,
            "y_bin": float(r["y_bin"]),
            "y_reg": float(r["y_reg"]) if "y_reg" in r else 0.0,
            "meta": {
                "gender": r.get("gender",""),
                "session": str(r.get("session","")),
                "dataset": r.get("dataset",""),
            }
        }
