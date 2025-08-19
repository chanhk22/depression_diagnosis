# train/train_multimodal.py (요약)
# 1) load datasets (E-DAIC train + D-VLOG train)
# 2) build model, optimizer, scheduler
# 3) training loop for epochs:
#    - for batch in dataloader:
#        logits = model(wave, lld, visual)
#        loss = BCEWithLogitsLoss(logits, labels)
#        if use_mmd: loss += lambda_mmd * mmd(feats_by_domain)
#        backward & step
#    - validate on both E-DAIC valid and D-VLOG valid
#    - save best checkpoints by avg AUROC
import yaml, pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.collate import collate_fn
from models.teacher_student import Teacher
from models.losses import BCEWithLogits
from utils.seed_utils import set_seed
from utils.metrics import compute_metrics
from pathlib import Path
from tqdm import tqdm

def run_epoch(model, loader, optim=None, device="cuda"):
    is_train = optim is not None
    model.train() if is_train else model.eval()
    y_true, y_prob = [], []
    loss_fn = BCEWithLogits()
    total_loss=0.0
    for batch in tqdm(loader, disable=False):
        lld = torch.tensor(batch["lld"], dtype=torch.float32, device=device)
        y   = torch.tensor(batch["label"], dtype=torch.float32, device=device)
        priv = batch.get("priv", None)
        if priv is not None:
            priv = torch.tensor(priv, dtype=torch.float32, device=device)
        else:
            # teacher는 특권 없으면 스킵
            continue
        if is_train: optim.zero_grad()
        logits,_ = model(lld, priv)
        loss = loss_fn(logits, y)
        if is_train:
            loss.backward(); optim.step()
        total_loss += loss.item()*y.size(0)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    avg_loss = total_loss/max(1,len(y_true))
    return avg_loss, compute_metrics(y_true, y_prob)

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    ap.add_argument("--index", default=None)  # override index.csv
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    set_seed(args.seed)
    env = yaml.safe_load(open(args.env))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_csv = args.index or (Path(env["paths"]["cache_root"]) / env["outputs"]["index_csv"])

    # Teacher는 특권 있는 edaic/daic만 훈련
    train_set = WindowDataset(index_csv, split="train", domains=["edaic","daic"])
    dev_set   = WindowDataset(index_csv, split="dev",   domains=["edaic","daic"])
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_set, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    # 특권 차원 추정(샘플에서)
    sample = next(iter(train_loader))
    priv_dim = sample["priv"].shape[-1]
    model = Teacher(d=256, priv_in=priv_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc=-1; ckpt_dir = Path(env["paths"]["out_root"]) / env["train"]["ckpt_dir"]; ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs+1):
        tr_loss, tr_met = run_epoch(model, train_loader, optim, device)
        dv_loss, dv_met = run_epoch(model, dev_loader, None, device)
        print(f"[Ep{ep}] Train AUC {tr_met['AUROC']:.3f} | Dev AUC {dv_met['AUROC']:.3f}")
        if dv_met["AUROC"] > best_auc:
            best_auc = dv_met["AUROC"]
            torch.save(model.state_dict(), ckpt_dir / "teacher_best.pt")
            print("Saved best teacher.")
