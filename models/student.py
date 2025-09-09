# models/student.py
import torch
import torch.nn as nn
from .fusion_blocks import CrossAttention


class Student(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden"]
        L = cfg["lstm_layers"]
        H = cfg["attn_heads"]
        dp = cfg["dropout"]
        
        self.use_landmarks = cfg.get("use_landmarks", False)
        
        # Audio encoder (always present)
        self.audio_enc = nn.LSTM(
            cfg["audio_dim"], h, L, 
            batch_first=True, bidirectional=True, 
            dropout=dp if L > 1 else 0
        )
        
        # Visual encoder (optional)
        if self.use_landmarks:
            self.vis_enc = nn.LSTM(
                cfg["vis_dim"], h, L, 
                batch_first=True, bidirectional=True, 
                dropout=dp if L > 1 else 0
            )
            self.cross_a_v = CrossAttention(h*2, heads=H, dropout=dp)
        
        # Output heads
        self.out_bin = nn.Linear(h*2, 1)
        self.out_reg = nn.Linear(h*2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dp)
    
    def forward(self, audio, vis=None):
        """
        Forward pass
        audio: (B, T, audio_dim) - required
        vis: (B, T, vis_dim) - optional visual features
        """
        # Encode audio
        Ha, _ = self.audio_enc(audio)  # (B, T, 2*hidden)
        
        # Fuse with visual features if available and enabled
        if self.use_landmarks and vis is not None:
            Hv, _ = self.vis_enc(vis)  # (B, T, 2*hidden)
            Ha = self.cross_a_v(Ha, Hv)  # (B, T, 2*hidden)
        
        # Global pooling
        h = Ha.mean(dim=1)  # (B, 2*hidden)
        h = self.dropout(h)
        
        # Output predictions
        yb = torch.sigmoid(self.out_bin(h).squeeze(-1))  # (B,)
        yr = self.out_reg(h).squeeze(-1)  # (B,)
        
        return yb, yr, h  # h for knowledge distillation