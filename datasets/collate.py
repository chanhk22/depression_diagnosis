# datasets/collate.py
import torch

def pad_and_stack(batch, key):
    lens = [b[key].shape[0] for b in batch]
    maxL = max(lens)
    feat_dim = batch[0][key].shape[-1]
    out = torch.zeros(len(batch), maxL, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), maxL, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b[key].shape[0]
        out[i, :L] = torch.from_numpy(b[key]).float()
        mask[i, :L] = 1
    return out, mask

def collate_fn(batch):
    x_a, m_a = pad_and_stack(batch, "audio")
    x_v, m_v = pad_and_stack(batch, "landmark")  # may be zeros for E-DAIC
    y = torch.tensor([b["label"] for b in batch], dtype=torch.float32)  # binary
    meta = [b["meta"] for b in batch]
    return {"audio": x_a, "audio_mask": m_a,
            "landmark": x_v, "landmark_mask": m_v,
            "y": y, "meta": meta}
