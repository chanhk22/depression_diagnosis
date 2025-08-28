import torch, torch.nn as nn
from .fusion_blocks import CrossAttention

class Student(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden"]; L = cfg["lstm_layers"]; H = cfg["attn_heads"]; dp = cfg["dropout"]
        self.use_landmarks = cfg.get("use_landmarks", False)
        self.audio_enc = nn.LSTM(cfg["audio_dim"], h, L, batch_first=True, bidirectional=True, dropout=dp)
        if self.use_landmarks:
            self.vis_enc = nn.LSTM(cfg["vis_dim"], h, L, batch_first=True, bidirectional=True, dropout=dp)
            self.cross_a_v = CrossAttention(h*2, heads=H, dropout=dp)
        self.out_bin = nn.Linear(h*2, 1)
        self.out_reg = nn.Linear(h*2, 1)

    def forward(self, audio, vis=None):
        Ha = self.audio_enc(audio)[0]
        if self.use_landmarks and vis is not None:
            Hv = self.vis_enc(vis)[0]
            Ha = self.cross_a_v(Ha, Hv)
        h = Ha.mean(1)
        yb = torch.sigmoid(self.out_bin(h))
        yr = self.out_reg(h)
        return yb, yr, h
