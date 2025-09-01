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

    @staticmethod
    def default_collate_fn(batch):
        """
        Collate function for multimodal windows
        Supports variable-length sequence batching (bucketing-style).
        """
        if not batch:
            return {}

        # Extract all samples
        samples = batch
        out = {
            "audio": [],
            "vis": [],
            "priv": {},
            "y_bin": [],
            "y_reg": [],
            "meta": []
        }

        priv_keys = ["mfcc", "vgg", "densenet", "aus", "openface"]
        for k in priv_keys:
            out["priv"][k] = []

        # Collect raw sequences
        audio_lens, vis_lens = [], []
        for s in samples:
            audio = s.get("audio")
            if audio is not None:
                out["audio"].append(torch.tensor(audio, dtype=torch.float32))
                audio_lens.append(audio.shape[0])
            else:
                out["audio"].append(torch.zeros(1, 25))
                audio_lens.append(1)

            vis = s.get("landmarks")
            if vis is not None:
                if vis.ndim == 3 and vis.shape[1] == 68 and vis.shape[2] == 2:
                    vis = vis.reshape(vis.shape[0], -1)
                out["vis"].append(torch.tensor(vis, dtype=torch.float32))
                vis_lens.append(vis.shape[0])
            else:
                out["vis"].append(None)
                vis_lens.append(0)

            for k in priv_keys:
                val = s.get(k)
                if val is not None:
                    if val.ndim == 1:
                        val = val.reshape(1, -1)
                    out["priv"][k].append(torch.tensor(val, dtype=torch.float32))
                else:
                    out["priv"][k].append(None)

            out["y_bin"].append(s.get("y_bin", 0.0))
            out["y_reg"].append(s.get("y_reg", 0.0))
            out["meta"].append(s.get("meta", {}))

        # Pad sequences (audio & vis)
        max_audio = max(audio_lens)
        audio_padded = []
        for a in out["audio"]:
            if a.shape[0] < max_audio:
                pad = torch.zeros(max_audio - a.shape[0], a.shape[1])
                a = torch.cat([a, pad], dim=0)
            audio_padded.append(a)
        out["audio"] = torch.stack(audio_padded)

        if any(v is not None for v in out["vis"]):
            max_vis = max(vis_lens)
            vis_padded = []
            for v in out["vis"]:
                if v is None:
                    vis_padded.append(torch.zeros(max_vis, 136))
                else:
                    if v.shape[0] < max_vis:
                        pad = torch.zeros(max_vis - v.shape[0], v.shape[1])
                        v = torch.cat([v, pad], dim=0)
                    vis_padded.append(v)
            out["vis"] = torch.stack(vis_padded)
        else:
            out["vis"] = None

        for k in priv_keys:
            vals = [p for p in out["priv"][k] if p is not None]
            if vals:
                out["priv"][k] = torch.stack(vals)
            else:
                out["priv"][k] = None

        out["y_bin"] = torch.tensor(out["y_bin"], dtype=torch.float32)
        out["y_reg"] = torch.tensor(out["y_reg"], dtype=torch.float32)

        return out


    