# models/heads.py
import torch.nn as nn, torch

class SessionPooling(nn.Module):
    def __init__(self, hid): super().__init__(); self.att = nn.Linear(hid, 1)
    def forward(self, F, mask=None):
        if mask is not None:
            w = torch.where(mask.unsqueeze(-1), self.att(F), torch.full_like(F[...,:1], -1e9))
        else:
            w = self.att(F)
        w = torch.softmax(w, dim=1)
        return (w * F).sum(1)   # (B,H)

class Heads(nn.Module):
    def __init__(self, hid=256):
        super().__init__()
        self.pool = SessionPooling(hid)
        self.cls = nn.Linear(hid, 1)
        self.reg = nn.Linear(hid, 1)

    def forward(self, F, mask=None):
        s = self.pool(F, mask)       # (B,H)
        logit = self.cls(s).squeeze(1)
        phq = self.reg(s).squeeze(1)
        return logit, phq
