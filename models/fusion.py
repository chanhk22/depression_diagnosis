# models/fusion.py
import torch, torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, hid=256, heads=4, dropout=0.1):
        super().__init__()
        self.attn_av = nn.MultiheadAttention(hid, heads, dropout=dropout, batch_first=True)
        self.attn_va = nn.MultiheadAttention(hid, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hid); self.ln2 = nn.LayerNorm(hid)
        self.ffn = nn.Sequential(nn.Linear(hid, 4*hid), nn.GELU(), nn.Linear(4*hid, hid))
        self.ln3 = nn.LayerNorm(hid)

    def forward(self, A, V, mA=None, mV=None):
        # A attends V and V attends A, then sum
        q1, _ = self.attn_av(A, V, V, key_padding_mask=(~mV) if mV is not None else None)
        q2, _ = self.attn_va(V, A, A, key_padding_mask=(~mA) if mA is not None else None)
        F = self.ln1(A + q1) + self.ln2(V + q2)   # (B,L,H)
        F = self.ln3(F + self.ffn(F))
        return F
