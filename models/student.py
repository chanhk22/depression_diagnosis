

class StudentModel(nn.Module):
    def __init__(self, audio_dim=25, lmk_dim=136, hid=256):
        super().__init__()
        self.aenc = AudioEncoder(audio_dim, hid)
        self.venc = LandmarkEncoder(lmk_dim, hid)
        self.fuse = CrossAttentionFusion(hid)
        self.cls = nn.Linear(hid*2, 1)
        self.reg = nn.Linear(hid*2, 1)
    def forward(self, audio, visual):
        A = self.aenc(audio); V = self.venc(visual)
        fused = self.fuse(A, V)
        return self.cls(fused).squeeze(-1), self.reg(fused).squeeze(-1), fused