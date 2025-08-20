# losses/kd.py
import torch.nn.functional as F, torch

def kd_loss(student_logits, teacher_logits, T=3.0, alpha=0.7, y_true=None):
    # soft
    p_s = F.log_softmax(student_logits/T, dim=-1)
    p_t = F.softmax(teacher_logits/T, dim=-1)
    kl = F.kl_div(p_s, p_t, reduction="batchmean") * (T*T)
    if y_true is None: return alpha*kl
    # hard
    ce = F.binary_cross_entropy_with_logits(student_logits.squeeze(-1), y_true.float())
    return alpha*kl + (1-alpha)*ce
