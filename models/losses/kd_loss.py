import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, T=3.0, alpha=0.7, y_true=None):
    # student_logits, teacher_logits: (B,)
    s = F.log_softmax(student_logits / T, dim=-1) if student_logits.dim()>1 else F.log_softmax(student_logits.unsqueeze(-1)/T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1) if teacher_logits.dim()>1 else F.softmax(teacher_logits.unsqueeze(-1)/T, dim=-1)
    kl = F.kl_div(s, t, reduction='batchmean') * (T*T)
    if y_true is None:
        return alpha * kl
    ce = F.binary_cross_entropy_with_logits(student_logits, y_true.float())
    return alpha * kl + (1-alpha) * ce
