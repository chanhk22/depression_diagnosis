import torch, torch.nn as nn
import torch.nn.functional as F

class BCEWithLogits(nn.Module):
    def __init__(self): super().__init__(); self.loss=nn.BCEWithLogitsLoss()
    def forward(self, logits, y): return self.loss(logits, y.float())

class KDLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.5):
        super().__init__(); self.T=T; self.alpha=alpha; self.bce=nn.BCEWithLogitsLoss()
    def forward(self, s_logits, t_logits, y):
        ce = self.bce(s_logits, y.float())
        if t_logits is None:
            return ce
        kd = F.kl_div(
            F.logsigmoid(s_logits/self.T), torch.sigmoid(t_logits/self.T),
            reduction='batchmean'
        ) * (self.T**2)
        return (1-self.alpha)*ce + self.alpha*kd

def mmd_loss(x, y):
    # x,y pooled (B,D)
    xx = x @ x.t(); yy = y @ y.t(); xy = x @ y.t()
    rx = xx.diag().unsqueeze(0); ry = yy.diag().unsqueeze(0)
    Kxx = torch.exp(- (rx.t() + rx - 2*xx))
    Kyy = torch.exp(- (ry.t() + ry - 2*yy))
    Kxy = torch.exp(- (rx.t() + ry - 2*xy))
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
