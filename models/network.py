# models/networks.py
import torch, torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, in_dim=25, hid=128):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, batch_first=True, bidirectional=True)
    def forward(self,x):
        h,_ = self.rnn(x)
        return h.mean(1)

class VisualEncoder(nn.Module):
    def __init__(self, in_dim=136, hid=128):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, batch_first=True, bidirectional=True)
    def forward(self,x):
        h,_ = self.rnn(x)
        return h.mean(1)

class MultiModalTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = AudioEncoder()
        self.v = VisualEncoder()
        self.fc = nn.Linear(256+256,1)
    def forward(self,audio,visual):
        ha=self.a(audio); hv=self.v(visual)
        return torch.sigmoid(self.fc(torch.cat([ha,hv],-1)))
