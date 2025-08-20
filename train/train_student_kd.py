# train/train_kd.py
import os, yaml, torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from utils.config import load_config
from utils.seed import set_seed
from datasets.collate import collate_fn
from datasets.edaic import EDAICDataset
from datasets.daic import DAICDataset
from datasets.dvlog import DVLOGDataset
from models.encoders import AudioLSTM, LandmarkCNNLSTM
from models.fusion import CrossAttentionFusion
from models.heads import Heads
from losses.kd import kd_loss
from losses.grl import grad_reverse
from losses.mmd import gaussian_mmd

def build_model(cfg):
    H = cfg["model"]["hidden"]
    enc_a = AudioLSTM(in_dim=cfg["data"]["lld_dim"], hid=H)
    enc_v = LandmarkCNNLSTM(in_dim=cfg["data"]["lmk_dim"], hid=H)
    fuse = CrossAttentionFusion(hid=H, heads=cfg["model"]["cross_attn_heads"], dropout=cfg["model"]["dropout"])
    heads = Heads(hid=H)
    domain_disc = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 3)) # DAIC/E-DAIC/D-VLOG
    return enc_a, enc_v, fuse, heads, domain_disc

def step(batch, modules, cfg, optimizer=None, teacher=None, domain=None):
    enc_a, enc_v, fuse, heads, disc = modules
    x_a, m_a = batch["audio"], batch["audio_mask"]
    x_v, m_v = batch["landmark"], batch["landmark_mask"]
    y = batch["y"]
    A = enc_a(x_a, m_a); V = enc_v(x_v, m_v)
    F = fuse(A, V, m_a, m_v)
    logit, phq = heads(F, m_a & m_v)
    loss = nn.functional.binary_cross_entropy_with_logits(logit, y)

    # KD (교사가 있을 때)
    if teacher is not None:
        with torch.no_grad():
            t_logit, _ = teacher_forward(teacher, x_a, m_a, x_v, m_v)
        loss = loss + kd_loss(logit.unsqueeze(-1), t_logit.unsqueeze(-1),
                              T=cfg["losses"]["kd"]["temp"],
                              alpha=cfg["losses"]["kd"]["alpha"],
                              y_true=y)

    # Domain Adapt (GRL)
    if cfg["losses"]["da"]["use_grl"] and domain is not None:
        F_rev = grad_reverse(F.mean(1), cfg["losses"]["da"]["grl_lambda"])
        d_logit = disc(F_rev)
        d_loss = nn.functional.cross_entropy(d_logit, domain)  # 0/1/2
        loss = loss + d_loss

    # MMD(Average pooled)
    if cfg["losses"]["da"]["use_mmd"]:
        # 간단히 배치 내에서 도메인별 평균 임베딩을 뽑아 mmd
        # 실제로는 소스/타깃 분리 필요
        pass

    if optimizer:
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(enc_a.parameters())+list(enc_v.parameters())+
                                 list(fuse.parameters())+list(heads.parameters()), cfg["training"]["grad_clip"])
        optimizer.step()
    with torch.no_grad():
        pred = (logit.sigmoid() > 0.5).float()
        acc = (pred == y).float().mean()
    return loss.item(), acc.item()

def teacher_forward(teacher, x_a, m_a, x_v, m_v):
    enc_a_t, enc_v_t, fuse_t, heads_t, _ = teacher
    A = enc_a_t(x_a, m_a)
    # 교사는 특권(예: AUs 등)을 enc_v_t의 입력으로 추가하거나 concat해서 구성 가능(여기선 단순히 동일 경로)
    V = enc_v_t(x_v, m_v)
    F = fuse_t(A, V, m_a, m_v)
    logit, phq = heads_t(F, m_a & m_v)
    return logit, phq

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    # Datasets
    ds_train = []
    if cfg["data"]["datasets"]["edaic"]["use"]: ds_train.append(EDAICDataset(cfg, "train"))
    if cfg["data"]["datasets"]["daic"]["use"]:  ds_train.append(DAICDataset(cfg, "train"))
    if cfg["data"]["datasets"]["dvlog"]["use"]: ds_train.append(DVLOGDataset(cfg, "train"))
    trainset = ConcatDataset(ds_train)
    loader = DataLoader(trainset, batch_size=cfg["training"]["batch_size"], shuffle=True,
                        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn, drop_last=True)
    # Model
    modules = build_model(cfg)
    params = []
    for m in modules: params += list(m.parameters())
    optim = torch.optim.Adam(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    # (옵션) 교사 로드 or 별도 사전학습
    teacher = None

    for epoch in range(cfg["training"]["max_epochs"]):
        logs = []
        for batch in loader:
            # domain id 만들기
            src = [m["src"] for m in batch["meta"]]
            dom = torch.tensor([{"DAIC":0,"E-DAIC":1,"D-VLOG":2}[s] for s in src])
            loss, acc = step(batch, modules, cfg, optimizer=optim, teacher=teacher, domain=dom)
            logs.append((loss, acc))
        print(f"epoch {epoch}: loss={sum(l for l,_ in logs)/len(logs):.4f} acc={sum(a for _,a in logs)/len(logs):.4f}")

if __name__ == "__main__":
    main()
