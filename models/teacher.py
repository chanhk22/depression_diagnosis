# models/teacher.py
import torch
import torch.nn as nn
from .fusion_blocks import CrossAttention


class Teacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden"]
        L = cfg["lstm_layers"]
        H = cfg["attn_heads"]
        dp = cfg["dropout"]
        
        # Audio encoder (always present)
        self.audio_enc = nn.LSTM(
            cfg["audio_dim"], h, L, 
            batch_first=True, bidirectional=True, dropout=dp if L > 1 else 0
        )
        
        # Visual encoder (landmarks)
        self.vis_enc = nn.LSTM(
            cfg["vis_dim"], h, L,
            batch_first=True, bidirectional=True, dropout=dp if L > 1 else 0
        )
        
        # Privileged feature projections
        self.vgg_fc = nn.Linear(cfg["vgg_dim"], h*2)
        self.densenet_fc = nn.Linear(cfg["densenet_dim"], h*2)
        self.face_fc = nn.Linear(cfg.get("face_feat_dim", 49), h*2)
        
        # Cross-attention modules
        self.cross_a_v = CrossAttention(h*2, heads=H, dropout=dp)
        self.cross_a_p = CrossAttention(h*2, heads=H, dropout=dp)
        
        # Output heads
        self.out_bin = nn.Linear(h*2, 1)
        self.out_reg = nn.Linear(h*2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dp)
        
    def encode_audio(self, a):
        """Encode audio features"""
        # a: (B, T, audio_dim)
        out, _ = self.audio_enc(a)
        return out  # (B, T, 2*hidden)
    
    def encode_vis(self, v):
        """Encode visual features (landmarks)"""
        # v: (B, T, vis_dim)
        out, _ = self.vis_enc(v)
        return out  # (B, T, 2*hidden)
    
    def fuse_priv(self, h, priv):
        """Fuse privileged features with main representation"""
        # h: (B, T, 2*hidden) - main audio/visual features
        # priv: dict with privileged features
        
        if priv is None:
            return h
            
        pools = []
        
        # Process each type of privileged feature
        if priv.get("vgg") is not None:
            vgg_feat = self.vgg_fc(priv["vgg"])  # (B, 1, 2*hidden) or (B, 2*hidden)
            if vgg_feat.dim() == 2:
                vgg_feat = vgg_feat.unsqueeze(1)  # (B, 1, 2*hidden)
            pools.append(vgg_feat)
            
        if priv.get("densenet") is not None:
            dense_feat = self.densenet_fc(priv["densenet"])
            if dense_feat.dim() == 2:
                dense_feat = dense_feat.unsqueeze(1)
            pools.append(dense_feat)
            
        if priv.get("face") is not None:
            face_feat = self.face_fc(priv["face"])
            if face_feat.dim() == 2:
                face_feat = face_feat.unsqueeze(1)
            pools.append(face_feat)
        
        if pools:
            # Average pooling of privileged features
            P = sum(pools) / len(pools)  # (B, 1, 2*hidden)
            
            # Cross-attention: audio attends to privileged features
            hp = self.cross_a_p(h, P)  # (B, T, 2*hidden)
            return hp
        else:
            return h
    
    def forward(self, audio, vis=None, priv=None):
        """
        Forward pass
        audio: (B, T, audio_dim) - required
        vis: (B, T, vis_dim) - optional visual features
        priv: dict - optional privileged features
        """
        # Encode audio (always present)
        Ha = self.encode_audio(audio)  # (B, T, 2*hidden)
        
        # Fuse with visual features if available
        if vis is not None:
            Hv = self.encode_vis(vis)  # (B, T, 2*hidden)
            # Audio attends to visual
            Ha = self.cross_a_v(Ha, Hv)  # (B, T, 2*hidden)
        
        # Fuse with privileged features if available
        if priv is not None:
            Ha = self.fuse_priv(Ha, priv)  # (B, T, 2*hidden)
        
        # Global pooling
        h = Ha.mean(dim=1)  # (B, 2*hidden)
        h = self.dropout(h)
        
        # Output predictions
        yb = torch.sigmoid(self.out_bin(h).squeeze(-1))  # (B,)
        yr = self.out_reg(h).squeeze(-1)  # (B,)
        
        return yb, yr, h  # h for knowledge distillation