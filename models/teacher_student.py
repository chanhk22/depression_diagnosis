import torch, torch.nn as nn
from .audio_enc_flex import AudioEncFlex
from .visual_enc import VisualEncoder
from .cross_attn import CrossAttn
from .fusion_transformer import FusionClassifier

class Teacher(nn.Module):
    # audio + privileged
    def __init__(self, d=256, priv_dim=64):
        super().__init__()
        self.aud = AudioEncFlex(use_pase=False)  # LLD only or add PASE if 원하면
        self.priv = nn.Sequential(nn.Linear(priv_dim, d), nn.ReLU(), nn.LayerNorm(d))
        self.cross = CrossAttn(d, 8)
        self.cls = FusionClassifier(input_dim=d*2, d_model=d)

    def forward(self, lld, priv, mask_a=None, mask_p=None):
        A = self.aud(lld)                # (B,T,256)
        P = self.priv(priv)              # (B,T,256)
        F = self.cross(A,P,mask_a,mask_p)# (B,T,512)
        return self.cls(F)               # (B,)

class Student(nn.Module):
    # audio (+ landmark when available)
    def __init__(self, d=256):
        super().__init__()
        self.aud = AudioEncFlex(use_pase=False)  # 통일성 위해 LLD-only 기본
        self.vis = VisualEncoder(in_dim=140, hid=d)
        self.cross = CrossAttn(d, 8)
        self.cls = FusionClassifier(input_dim=d*2, d_model=d)

    def forward(self, lld, lmk=None, micro=None, domain_idx=0, mask_a=None, mask_v=None):
        A = self.aud(lld)  # (B,T,256)
        if lmk is None:
            # 시각 모달 없음(E-DAIC) → zero & mask False
            V = torch.zeros_like(A)
            mask_v = torch.zeros(A.shape[:2], dtype=torch.bool, device=A.device)
        else:
            if micro is not None:
                x = torch.cat([lmk, micro], -1)  # (B,T,140)
            else:
                x = lmk
            V = self.vis(x, domain_idx, mask_v)  # (B,T,256)
        F = self.cross(A,V,mask_a,mask_v)        # (B,T,512)
        return self.cls(F)                       # (B,)
