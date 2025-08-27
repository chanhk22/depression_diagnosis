class CrossAttentionFusion(nn.Module):
    def __init__(self, hid=256, heads=4):
        super().__init__()
        self.att_av = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.att_va = nn.MultiheadAttention(hid, heads, batch_first=True)
    def forward(self,a,v):
        a2v, _ = self.att_av(a, v, v)
        v2a, _ = self.att_va(v, a, a)
        # pool
        a_pool = a2v.mean(1); v_pool = v2a.mean(1)
        return torch.cat([a_pool, v_pool], dim=-1)