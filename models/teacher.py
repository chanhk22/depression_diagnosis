class TeacherModel(nn.Module):
    def __init__(self, audio_dim=25, lmk_dim=136, hid=256, priv_dim=None):
        super().__init__()
        self.aenc = AudioEncoder(audio_dim, hid)
        self.venc = LandmarkEncoder(lmk_dim, hid)
        self.fuse = CrossAttentionFusion(hid)
        self.priv_proj = nn.Linear(priv_dim, hid) if priv_dim else None
        self.cls = nn.Linear(hid*2, 1)
        self.reg = nn.Linear(hid*2, 1)
    def forward(self, audio, visual, priv=None):
        A = self.aenc(audio); V = self.venc(visual)
        fused = self.fuse(A, V)  # (B, 2H)
        if self.priv_proj is not None and priv is not None:
            fused = fused + self.priv_proj(priv)
        logit = self.cls(fused).squeeze(-1)
        phq = self.reg(fused).squeeze(-1)
        return logit, phq, fused