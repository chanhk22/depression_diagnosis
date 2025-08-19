import torch, torch.nn as nn

class DSBN1D(nn.Module):
    def __init__(self, num_features, num_domains=3):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(num_domains)])
    def forward(self, x, d):
        # x: (B,T,C) -> (B,C,T) for BN; d: domain index {0:EDAIC,1:DAIC,2:DVLOG}
        x = x.permute(0,2,1)
        x = self.bns[d](x)
        return x.permute(0,2,1)

class VisualEncoder(nn.Module):
    def __init__(self, in_dim=140, hid=256, layers=2, use_dsbn=True, num_domains=3):
        super().__init__()
        self.use_dsbn = use_dsbn
        self.rnn = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid*2, hid)
        if use_dsbn:
            self.dsbn = DSBN1D(hid, num_domains)

    def forward(self, x, domain_idx=0, mask=None):
        # x: (B,T,140). mask: (B,T) for time masking(optional)
        out,_ = self.rnn(x)
        h = self.proj(out)  # (B,T,256)
        if self.use_dsbn:
            h = self.dsbn(h, domain_idx)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h
