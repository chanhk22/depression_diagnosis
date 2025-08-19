import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    """
    Knowledge Distillation loss (LUPI).
    Combines CE + KL divergence between teacher & student logits.
    """
    def __init__(self, T=2.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce(student_logits, labels.float())
        if teacher_logits is not None:
            kd_loss = F.kl_div(
                F.logsigmoid(student_logits / self.T),
                torch.sigmoid(teacher_logits / self.T),
                reduction="batchmean"
            ) * (self.T ** 2)
            return (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        else:
            return ce_loss
