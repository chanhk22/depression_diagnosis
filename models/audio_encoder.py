import torch, torch.nn as nn

class AudioEncFlex(nn.Module):
    """
    항상 LLD(25 LLD) 사용 + (옵션) PASE/wav2vec 임베딩. D-VLOG는 LLD만.
    """
    def __init__(self, lld_dim=25, pase_dim=512, out_dim=256, use_pase=True):
        super().__init__()
        self.use_pase = use_pase
        if use_pase:
            # external pase model 주입 예정 -> forward에 pase_feats로 받음
            self.pase_proj = nn.Sequential(nn.Linear(pase_dim, out_dim//2), nn.ReLU(), nn.LayerNorm(out_dim//2))
        self.lld_proj  = nn.Sequential(nn.Linear(lld_dim, out_dim//2 if use_pase else out_dim),
                                       nn.ReLU(), nn.LayerNorm(out_dim//2 if use_pase else out_dim))
        self.out = nn.Linear(out_dim, out_dim)

    def forward(self, lld, pase_feats=None):
        # lld: (B,T,25), pase_feats: (B,T,pase_dim) or None
        l = self.lld_proj(lld)
        if self.use_pase and (pase_feats is not None):
            p = self.pase_proj(pase_feats)
            x = torch.cat([p, l], -1)
        else:
            x = l
        return self.out(x)  # (B,T,out_dim)
