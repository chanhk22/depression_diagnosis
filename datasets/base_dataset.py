import torch
import numpy as np

GENDER_MAP = {"M":0, "F":1, "m":0, "f":1}

def to_tensor(x, dtype=torch.float32):
    if x is None: return None
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    return x.to(dtype)

def default_collate_fn(batch):
    # batch: list of dicts with keys: audio, vis, priv, y_bin, y_reg, meta
    out = {}
    keys = ["audio","vis"]
    for k in keys:
        xs = [b[k] for b in batch if b[k] is not None]
        out[k] = torch.nn.utils.rnn.pad_sequence(
            [to_tensor(x) for x in xs], batch_first=True, padding_value=0.0
        ) if len(xs)>0 else None

    # privileged (optional, time-agnostic 1Hz)
    priv = {}
    for pk in ["vgg","densenet","aus"]:
        xs = [b["priv"].get(pk) for b in batch] if "priv" in batch[0] else []
        xs = [to_tensor(x) for x in xs if x is not None]
        priv[pk] = torch.stack(xs, dim=0) if xs else None
    out["priv"] = priv

    # labels
    out["y_bin"] = torch.tensor([b["y_bin"] for b in batch], dtype=torch.float32).unsqueeze(1)
    out["y_reg"] = torch.tensor([b["y_reg"] for b in batch], dtype=torch.float32).unsqueeze(1)

    # meta (gender embedding, session, dataset)
    genders = [GENDER_MAP.get(str(b["meta"].get("gender","")).strip(), -1) for b in batch]
    out["gender"] = torch.tensor(genders, dtype=torch.int64)
    out["session"] = [b["meta"].get("session") for b in batch]
    out["dataset"] = [b["meta"].get("dataset") for b in batch]
    return out
