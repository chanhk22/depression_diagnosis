import torch, torch.nn as nn

class CrossAttn(nn.Module):
    def __init__(self, d=256, heads=8):
        super().__init__()
        self.a2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.v2a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln  = nn.LayerNorm(d)

    def forward(self, A, V, mask_a=None, mask_v=None):
        # A,V: (B,T,D); mask_*: (B,T) -> True for keep
        key_padding_v = None if mask_v is None else ~mask_v.bool()
        key_padding_a = None if mask_a is None else ~mask_a.bool()
        a2v,_ = self.a2v(A, V, V, key_padding_mask=key_padding_v)
        v2a,_ = self.v2a(V, A, A, key_padding_mask=key_padding_a)
        A = self.ln(A + a2v)
        V = self.ln(V + v2a)
        return torch.cat([A,V], -1)  # (B,T,2D)
