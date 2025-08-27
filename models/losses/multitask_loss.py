# classification + regression joint loss

# losses/multitask_loss.py
import torch
import torch.nn.functional as F

def multitask_loss(logits, phq_pred, labels_binary=None, phq_targets=None, lambda_reg=0.3):
    """
    logits : (B,) raw logits for binary classification
    phq_pred : (B,) predicted PHQ score (regression)
    labels_binary : (B,) 0/1
    phq_targets : (B,) floats
    lambda_reg : weight for regression loss
    """
    loss = 0.0
    if labels_binary is not None:
        loss += F.binary_cross_entropy_with_logits(logits, labels_binary.float())
    if phq_targets is not None:
        loss += lambda_reg * F.mse_loss(phq_pred, phq_targets.float())
    return loss
