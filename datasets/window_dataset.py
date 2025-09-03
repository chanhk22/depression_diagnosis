import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

def _pool_to_step1(arr):
    """(T,D) -> (1,D) 평균 pooling. None이면 None."""
    if arr is None: return None
    a = np.asarray(arr)
    if a.ndim == 1: a = a[None, :]
    if a.ndim == 3:  # landmarks (T,68,2) -> (T,136) 후 평균
        T, L, _ = a.shape
        a = a.reshape(T, -1)
    return a.mean(axis=0, keepdims=True).astype(np.float32)

class WindowDataset(Dataset):
    """
    index_csv produced by build_windows contains 'path' column pointing to per-window .npz
    .npz keys: audio (T,25), landmarks (T,68,2), face_feat (T,~49), vgg (T,4096), densenet (T,1920) ...
    Returns:
      audio: (T,25) float32
      vis:   (T,136) or None
      priv:  dict -> 'face','vgg','densenet' pooled to (1,D) tensors or None
      y_bin/y_reg: optional (없으면 None)
      meta: dict
    """
    def __init__(self, index_csv, modal_keys=("audio","landmarks","mfcc","face_feat","vgg","densenet")):
        self.df = pd.read_csv(index_csv)
        self.modal_keys = modal_keys

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = r["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        npz = np.load(p, allow_pickle=True)

        # audio
        audio = npz["audio"].astype(np.float32) if "audio" in npz else None

        # vis = landmarks
        vis = None
        if "landmarks" in npz:
            lm = np.asarray(npz["landmarks"])
            if lm.ndim == 3 and lm.shape[1]==68 and lm.shape[2]==2:
                vis = lm.reshape(lm.shape[0], -1).astype(np.float32)
            elif lm.ndim == 2 and lm.shape[1]==136:
                vis = lm.astype(np.float32)

        # priv (pooled to 1-step)
        priv = {}
        face = _pool_to_step1(npz["face_feat"]) if "face_feat" in npz else None
        vgg  = _pool_to_step1(npz["vgg"])       if "vgg" in npz else None
        den  = _pool_to_step1(npz["densenet"])  if "densenet" in npz else None
        if face is not None: priv["face"] = face
        if vgg  is not None: priv["vgg"] = vgg
        if den  is not None: priv["densenet"] = den

        sample = {
            "audio": audio,
            "vis": vis,
            "priv": priv,
            "y_bin": float(r["y_bin"]) if "y_bin" in r and not pd.isna(r["y_bin"]) else None,
            "y_reg": float(r["y_reg"]) if "y_reg" in r and not pd.isna(r["y_reg"]) else None,
            "meta": {"session": r.get("session"), "t0": r.get("t0"), "t1": r.get("t1")}
        }
        return sample
