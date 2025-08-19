import torch, torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=4, nhead=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over time
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.transformer(h) # (B, T, d_model)
        # pool time dim
        h_t = h.permute(0,2,1)  # (B, d_model, T)
        p = self.pool(h_t).squeeze(-1)  # (B, d_model)
        logits = self.classifier(p).squeeze(-1)
        return logits
