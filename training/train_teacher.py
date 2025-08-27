# train/train_full_pipeline.py
import os, yaml, random, numpy as np, torch
from precache.build_windows import build_session_windows
from datasets.window_dataset import WindowDataset
from models.architectures import TeacherModel, StudentModel
from models.losses import kd_loss, gaussian_mmd, grad_reverse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def train_teacher(cfg):
    # build dataset from cache (user must have built .npz & index.csv)
    index = os.path.join(cfg['paths']['cache_root'], 'combined_index_teacher.csv')
    ds = WindowDataset(index)  # user should prepare combined index from DAIC+E-DAIC
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    device = torch.device(cfg['training']['device'])
    teacher = TeacherModel(audio_dim=cfg['features']['lld_dim'], lmk_dim=cfg['features']['lmk_dim'], hid=256, priv_dim=128).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=cfg['training']['lr'])
    for ep in range(cfg['training']['max_epochs_teacher']):
        teacher.train()
        losses = []
        for b in dl:
            a = b['audio'].to(device).float()
            v = b['visual'].to(device).float()
            # dummy privileged (implement loading if exists)
            priv = None
            y = torch.zeros(a.size(0), device=device)  # placeholder: load real labels
            logit, phq, _ = teacher(a, v, priv)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + 0.3 * torch.nn.functional.mse_loss(phq, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Teacher epoch {ep} loss {np.mean(losses):.4f}")
    os.makedirs(cfg['paths']['out_root'], exist_ok=True)
    torch.save(teacher.state_dict(), os.path.join(cfg['paths']['out_root'], 'teacher.pt'))
    return teacher