# losses/kd_loss.py
import torch
import torch.nn.functional as F

def knowledge_distillation_loss(student_logits, teacher_logits, labels=None, T=3.0, alpha=0.7):
    """
    student_logits, teacher_logits: (B,) or (B,C) - for binary we treat as (B,1)
    labels: ground-truth binary labels or None
    returns combined loss = alpha * KL( soft_teacher || soft_student ) + (1-alpha) * CE(student, labels)
    """
    # make (B,2) if binary logits single dimension: create two-class logits [ -logit, logit ]? but easiest is to use sigmoid-based KD via probabilities
    if student_logits.dim() == 1 or student_logits.shape[1] == 1:
        # convert to probs with sigmoid
        s_prob = torch.sigmoid(student_logits)
        t_prob = torch.sigmoid(teacher_logits)
        # KL between Bernoulli distributions: use KLDiv on log probs?
        # We'll use MSE on logits (soft target regression) + CE for hard labels for stability
        loss_kd = F.mse_loss(s_prob, t_prob)
    else:
        s_log = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(s_log, t_soft, reduction='batchmean') * (T * T)

    if labels is None:
        return alpha * loss_kd
    else:
        ce = F.binary_cross_entropy_with_logits(student_logits.squeeze(-1), labels.float())
        return alpha * loss_kd + (1.0 - alpha) * ce
