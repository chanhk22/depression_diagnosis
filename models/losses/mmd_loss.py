import torch

def gaussian_mmd(x, y, sigma_list=[1,2,4,8,16]):
    # x,y: (N,D)
    # simple implementation
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())
    rx = (x*x).sum(1).unsqueeze(1)
    ry = (y*y).sum(1).unsqueeze(1)
    Kxx = torch.exp(- (rx - 2*xx + rx.t()) / (2 * sigma_list[0]**2))
    Kyy = torch.exp(- (ry - 2*yy + ry.t()) / (2 * sigma_list[0]**2))
    Kxy = torch.exp(- (rx - 2*xy + ry.t()) / (2 * sigma_list[0]**2))
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()