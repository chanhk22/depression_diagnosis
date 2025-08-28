import torch, torch.nn as nn, torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x, ctx):
        # x: (B,T,D)   ctx: (B,Tc,D)
        res = x
        out,_ = self.attn(self.ln(x), self.ln(ctx), self.ln(ctx))
        x = res + out
        x = x + self.ff(self.ln(x))
        return x

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd*grad_output, None
def grad_reverse(x, lambd=1.0): return GRL.apply(x, lambd)

def pairwise_mmd(x, y, kernel='rbf', sigma=1.0):
    # x: (B,D), y: (B,D)
    def rbf(a,b):
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True)
        dist = a2 + b2.T - 2*a@b.T
        k = torch.exp(-dist/(2*sigma*sigma))
        return k
    Kxx = rbf(x,y=x)
    Kyy = rbf(y,y=y)
    Kxy = rbf(x,y)
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
