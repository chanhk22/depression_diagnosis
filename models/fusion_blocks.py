# models/fusion_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class CrossAttention(nn.Module):
    """Cross-attention mechanism for multimodal fusion"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim) 
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key_value):
        """
        query: (B, T_q, dim) - what we want to update
        key_value: (B, T_kv, dim) - what we attend to
        """
        B, T_q, _ = query.shape
        _, T_kv, _ = key_value.shape
        
        # Project to query, key, value
        Q = self.q_proj(query)  # (B, T_q, dim)
        K = self.k_proj(key_value)  # (B, T_kv, dim)
        V = self.v_proj(key_value)  # (B, T_kv, dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T_q, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T_q, head_dim)
        K = K.view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T_kv, head_dim)
        V = V.view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T_kv, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, T_q, T_kv)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, heads, T_q, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.dim)  # (B, T_q, dim)
        
        # Final projection
        out = self.out_proj(out)
        
        # Residual connection
        return query + out


class GradientReversalFunction(Function):
    """Gradient Reversal Layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    """Apply gradient reversal"""
    return GradientReversalFunction.apply(x, lambda_)


def pairwise_mmd(x, y, sigma=1.0):
    """
    Maximum Mean Discrepancy between two distributions
    x, y: (B, D) tensors
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device)
    
    # Compute pairwise distances
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    # Diagonal elements
    xx_diag = torch.diag(xx).unsqueeze(1)
    yy_diag = torch.diag(yy).unsqueeze(1)
    
    # Distance matrices
    xx_dist = xx_diag + xx_diag.t() - 2 * xx
    yy_dist = yy_diag + yy_diag.t() - 2 * yy
    xy_dist = xx_diag + yy_diag.t() - 2 * xy
    
    # RBF kernel
    xx_k = torch.exp(-xx_dist / (2 * sigma ** 2))
    yy_k = torch.exp(-yy_dist / (2 * sigma ** 2))
    xy_k = torch.exp(-xy_dist / (2 * sigma ** 2))
    
    # MMD statistic
    mmd = xx_k.mean() + yy_k.mean() - 2 * xy_k.mean()
    
    return mmd


class TransformerLayer(nn.Module):
    """Transformer layer for sequence modeling"""
    def __init__(self, dim, heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
            
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, src_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class ModalityFusion(nn.Module):
    """Advanced multimodal fusion module"""
    def __init__(self, dim, fusion_type="concat", dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.dim = dim
        
        if fusion_type == "concat":
            # Simple concatenation + projection
            self.proj = nn.Linear(dim * 2, dim)
        elif fusion_type == "add":
            # Element-wise addition (requires same dim)
            pass
        elif fusion_type == "cross_attn":
            # Cross-attention based fusion
            self.cross_attn = CrossAttention(dim, dropout=dropout)
        elif fusion_type == "transformer":
            # Transformer-based fusion
            self.transformer = TransformerLayer(dim, dropout=dropout)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        """
        Fuse two modalities
        x1, x2: (B, T, dim) or (B, dim)
        """
        if self.fusion_type == "concat":
            if x1.dim() == 3 and x2.dim() == 3:
                # Sequence data: pool first
                x1_pooled = x1.mean(dim=1)  # (B, dim)
                x2_pooled = x2.mean(dim=1)  # (B, dim)
                fused = torch.cat([x1_pooled, x2_pooled], dim=1)  # (B, 2*dim)
                return self.dropout(self.proj(fused))  # (B, dim)
            else:
                # Already pooled
                fused = torch.cat([x1, x2], dim=1)
                return self.dropout(self.proj(fused))
                
        elif self.fusion_type == "add":
            if x1.dim() == 3 and x2.dim() == 3:
                return x1 + x2  # Element-wise addition
            else:
                return x1 + x2
                
        elif self.fusion_type == "cross_attn":
            if x1.dim() == 3:
                return self.cross_attn(x1, x2)
            else:
                # Add time dimension for cross-attention
                x1_seq = x1.unsqueeze(1)  # (B, 1, dim)
                x2_seq = x2.unsqueeze(1)  # (B, 1, dim)
                fused = self.cross_attn(x1_seq, x2_seq)  # (B, 1, dim)
                return fused.squeeze(1)  # (B, dim)
                
        elif self.fusion_type == "transformer":
            if x1.dim() == 3 and x2.dim() == 3:
                # Concatenate along sequence dimension
                combined = torch.cat([x1, x2], dim=1)  # (B, T1+T2, dim)
                fused = self.transformer(combined)  # (B, T1+T2, dim)
                return fused.mean(dim=1)  # (B, dim)
            else:
                # Stack and process
                stacked = torch.stack([x1, x2], dim=1)  # (B, 2, dim)
                fused = self.transformer(stacked)  # (B, 2, dim)
                return fused.mean(dim=1)  # (B, dim)