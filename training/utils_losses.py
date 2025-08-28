# training/utils_losses.py  (새 파일; 루프에서 한 줄로 호출)
import torch, torch.nn.functional as F
from models.fusion_blocks import pairwise_mmd, grad_reverse

def multitask_loss(yb, yb_tgt, yr, yr_tgt, reg_lambda=0.3):
    bce = F.binary_cross_entropy(yb, yb_tgt)
    mse = F.mse_loss(yr, yr_tgt)
    return bce + reg_lambda*mse, {"bce":bce.item(), "mse":mse.item()}

def kd_loss(student_logits, teacher_logits, T=2.0):
    # logits → soft targets
    s = torch.log(student_logits + 1e-8)
    t = torch.log(teacher_logits + 1e-8)
    return F.kl_div(s/T, (teacher_logits/T), reduction="batchmean") * (T*T)

def domain_losses(h_src, h_tgt, grl_lambda=0.5, mmd_lambda=0.5, use_grl=True, use_mmd=True):
    # h_*: (B,D)
    loss = 0.0; logs={}
    if use_mmd:
        mmd = pairwise_mmd(h_src, h_tgt, sigma=1.0)
        loss += mmd_lambda * mmd
        logs["mmd"] = float(mmd.item())
    if use_grl:
        # simple domain linear head
        D = h_src.size(1)
        # NOTE: 헤드는 외부에서 모듈로 주입하는 게 일반적이지만 여기선 간단히 inline
        W = torch.nn.Linear(D, 2).to(h_src.device)
        ds = W(grad_reverse(h_src, grl_lambda))
        dt = W(grad_reverse(h_tgt, grl_lambda))
        y_s = torch.zeros(ds.size(0), dtype=torch.long, device=ds.device)
        y_t = torch.ones(dt.size(0), dtype=torch.long, device=dt.device)
        ce = F.cross_entropy(torch.cat([ds,dt],0), torch.cat([y_s,y_t],0))
        loss += ce
        logs["grl_ce"] = float(ce.item())
    return loss, logs
