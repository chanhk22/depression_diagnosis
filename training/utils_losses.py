# training/utils_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion_blocks import pairwise_mmd, grad_reverse


def multitask_loss(yb, yb_tgt, yr, yr_tgt, reg_lambda=0.3):
    """
    Multitask loss combining binary classification and regression
    
    Args:
        yb: Binary predictions (B,)
        yb_tgt: Binary targets (B,)
        yr: Regression predictions (B,)
        yr_tgt: Regression targets (B,)
        reg_lambda: Weight for regression loss
    """
    # Ensure proper shapes
    yb = yb.squeeze() if yb.dim() > 1 else yb
    yr = yr.squeeze() if yr.dim() > 1 else yr
    yb_tgt = yb_tgt.squeeze() if yb_tgt.dim() > 1 else yb_tgt
    yr_tgt = yr_tgt.squeeze() if yr_tgt.dim() > 1 else yr_tgt
    
    # Binary cross-entropy loss
    bce = F.binary_cross_entropy(yb, yb_tgt, reduction='mean')
    
    # Mean squared error for regression
    mse = F.mse_loss(yr, yr_tgt, reduction='mean')
    
    # Combined loss
    total_loss = bce + reg_lambda * mse
    
    return total_loss, {"bce": bce.item(), "mse": mse.item()}


def kd_loss(student_probs, teacher_probs, T=2.0):
    """
    Knowledge Distillation loss using KL divergence
    
    Args:
        student_probs: Student probabilities (B,)
        teacher_probs: Teacher probabilities (B,)
        T: Temperature for softening distributions
    """
    # Ensure proper shapes
    student_probs = student_probs.squeeze() if student_probs.dim() > 1 else student_probs
    teacher_probs = teacher_probs.squeeze() if teacher_probs.dim() > 1 else teacher_probs
    
    # Clamp to avoid log(0)
    student_probs = torch.clamp(student_probs, min=1e-8, max=1-1e-8)
    teacher_probs = torch.clamp(teacher_probs, min=1e-8, max=1-1e-8)
    
    # For binary classification, convert to 2-class distributions
    student_dist = torch.stack([1-student_probs, student_probs], dim=1)  # (B, 2)
    teacher_dist = torch.stack([1-teacher_probs, teacher_probs], dim=1)  # (B, 2)
    
    # Apply temperature softening
    student_soft = F.log_softmax(torch.log(student_dist) / T, dim=1)
    teacher_soft = F.softmax(torch.log(teacher_dist) / T, dim=1)
    
    # KL divergence loss
    kd_loss_val = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
    
    return kd_loss_val


class DomainClassifier(nn.Module):
    """Separate domain classifier for GRL"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)


def domain_losses(h_src, h_tgt, domain_classifier=None, grl_lambda=0.5, mmd_lambda=0.5, 
                 use_grl=True, use_mmd=True):
    """
    Domain adaptation losses (MMD + GRL)
    
    Args:
        h_src: Source domain features (B, D)
        h_tgt: Target domain features (B, D)
        domain_classifier: Optional pre-initialized domain classifier
        grl_lambda: Weight for gradient reversal loss
        mmd_lambda: Weight for MMD loss
        use_grl: Whether to use gradient reversal loss
        use_mmd: Whether to use MMD loss
    """
    loss = 0.0
    logs = {}
    
    # MMD Loss
    if use_mmd and h_src.size(0) > 0 and h_tgt.size(0) > 0:
        try:
            mmd = pairwise_mmd(h_src, h_tgt, sigma=1.0)
            loss += mmd_lambda * mmd
            logs["mmd"] = float(mmd.item())
        except Exception as e:
            print(f"MMD computation failed: {e}")
            logs["mmd"] = 0.0
    
    # Gradient Reversal Loss
    if use_grl and h_src.size(0) > 0 and h_tgt.size(0) > 0:
        try:
            # Create or use provided domain classifier
            if domain_classifier is None:
                D = h_src.size(1)
                domain_classifier = nn.Linear(D, 2).to(h_src.device)
                # Initialize weights
                nn.init.xavier_uniform_(domain_classifier.weight)
                nn.init.zeros_(domain_classifier.bias)
            
            # Apply gradient reversal and classify
            h_src_rev = grad_reverse(h_src, grl_lambda)
            h_tgt_rev = grad_reverse(h_tgt, grl_lambda)
            
            # Domain predictions
            d_src = domain_classifier(h_src_rev)  # (B_src, 2)
            d_tgt = domain_classifier(h_tgt_rev)  # (B_tgt, 2)
            
            # Domain labels (0 for source, 1 for target)
            y_src = torch.zeros(d_src.size(0), dtype=torch.long, device=d_src.device)
            y_tgt = torch.ones(d_tgt.size(0), dtype=torch.long, device=d_tgt.device)
            
            # Combine predictions and labels
            d_combined = torch.cat([d_src, d_tgt], dim=0)
            y_combined = torch.cat([y_src, y_tgt], dim=0)
            
            # Cross-entropy loss for domain classification
            grl_loss = F.cross_entropy(d_combined, y_combined)
            loss += grl_loss
            logs["grl_ce"] = float(grl_loss.item())
            
            # Domain accuracy for monitoring
            with torch.no_grad():
                d_pred = torch.argmax(d_combined, dim=1)
                d_acc = (d_pred == y_combined).float().mean()
                logs["domain_acc"] = float(d_acc.item())
                
        except Exception as e:
            print(f"GRL computation failed: {e}")
            logs["grl_ce"] = 0.0
            logs["domain_acc"] = 0.5
    
    return loss, logs


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP (if needed)"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Get critic scores
    d_interpolates = critic(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def focal_loss(inputs, targets, alpha=1.0, gamma=2.0, reduction='mean'):
    """
    Focal Loss for handling class imbalance
    
    Args:
        inputs: Predictions (B,)
        targets: Ground truth (B,)
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val