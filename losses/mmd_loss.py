import torch
import torch.nn as nn
import torch.nn.functional as F

def mmd_loss(x, y, sigma=1.0):
    """
    Maximum Mean Discrepancy (RBF kernel).
    x: (N,D) from domain A
    y: (M,D) from domain B
    """
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    Kxx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
    Kyy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
    Kxy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * sigma ** 2))

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()