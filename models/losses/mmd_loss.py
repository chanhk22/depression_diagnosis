# losses/mmd_loss.py
import torch

def gaussian_kernel(x, y, sigma):
    # x: (n,d), y: (m,d)
    xx = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    return torch.exp(-xx / (2.0 * sigma * sigma))

def mmd_rbf(x, y, sigma_list=[1,2,4,8,16]):
    K = 0.0
    for s in sigma_list:
        K += gaussian_kernel(x, x, s).mean() + gaussian_kernel(y, y, s).mean() - 2.0 * gaussian_kernel(x, y, s).mean()
    return K

def mmd_loss(source_feats, target_feats, sigma_list=[1,2,4,8,16]):
    """
    source_feats: (n,d), target_feats: (m,d)
    returns scalar MMD
    """
    return mmd_rbf(source_feats, target_feats, sigma_list)
