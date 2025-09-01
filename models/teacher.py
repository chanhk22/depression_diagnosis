import torch, torch.nn as nn
from .fusion_blocks import CrossAttention

class Teacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden"]; L = cfg["lstm_layers"]; H = cfg["attn_heads"]; dp = cfg["dropout"]
        self.audio_enc = nn.LSTM(cfg["audio_dim"], h, L, batch_first=True, bidirectional=True, dropout=dp)
        self.vis_enc   = nn.LSTM(cfg["landmarks_dim"],   h, L, batch_first=True, bidirectional=True, dropout=dp)
        self.vgg_fc    = nn.Linear(cfg["vgg_dim"], h*2)
        self.dense_fc  = nn.Linear(cfg["densenet_dim"], h*2)
        self.face_fc    = nn.Linear(cfg.get("face_feat_dim",49), h*2)

        self.cross_a_v = CrossAttention(h*2, heads=H, dropout=dp)
        self.cross_a_p = CrossAttention(h*2, heads=H, dropout=dp)
        self.out_bin   = nn.Linear(h*2, 1)
        self.out_reg   = nn.Linear(h*2, 1)

    def encode_audio(self, a): return self.audio_enc(a)[0]     # (B,T,2h)
    def encode_vis(self, v):   return self.vis_enc(v)[0]       # (B,T,2h)

    def fuse_priv(self, h, priv):
        # h: (B,T,2h) pooled later; priv: dict with (B,1,D)
        hp = h
        pools=[]
        if priv.get("vgg") is not None: pools.append(self.vgg_fc(priv["vgg"]))
        if priv.get("densenet") is not None: pools.append(self.dense_fc(priv["densenet"]))
        if priv.get("aus") is not None: pools.append(self.aus_fc(priv["aus"]))
        if pools:
            P = sum(pools)/len(pools)         # (B,1,2h)
            hp = self.cross_a_p(h, P)         # attend from audio to priv
        return hp

    def forward(self, audio, vis=None, priv=None):
        Ha = self.encode_audio(audio)                 # (B,T,2h)
        if vis is not None:
            Hv = self.encode_vis(vis)
            Ha = self.cross_a_v(Ha, Hv)              # audio attends to visual
        if priv is not None:
            Ha = self.fuse_priv(Ha, priv)
        h = Ha.mean(dim=1)                            # global pooling
        yb = torch.sigmoid(self.out_bin(h))
        yr = self.out_reg(h)
        return yb, yr, h                               # h for KD/MMD
