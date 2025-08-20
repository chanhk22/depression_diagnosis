# losses/mmd.py
import torch

def gaussian_mmd(x, y, sigma=1.0):
    def pdist(a): return torch.cdist(a, a, p=2)**2
    Kxx = torch.exp(-pdist(x)/ (2*sigma**2)).mean()
    Kyy = torch.exp(-pdist(y)/ (2*sigma**2)).mean()
    Kxy = torch.exp(-torch.cdist(x, y, p=2)**2 / (2*sigma**2)).mean()
    return Kxx + Kyy - 2*Kxy
