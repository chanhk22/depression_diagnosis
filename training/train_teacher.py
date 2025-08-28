import os, yaml, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.base_dataset import default_collate_fn
from models.teacher import Teacher
from training.utils_losses import multitask_loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def eval_epoch(model, loader, device, th=0.5):
    model.eval()
    y_true=[]; y_prob=[]; y_reg=[]; y_pred=[]
    with torch.no_grad():
        for b in loader:
            a = b["audio"].to(device)
            v = b["vis"].to(device) if b["vis"] is not None else None
            priv = {k: (b["priv"][k].to(device) if b["priv"][k] is not None else None) for k in b["priv"]}
            yb, yr, _ = model(a, v, priv)
            y_true.extend(b["y_bin"].cpu().numpy().ravel().tolist())
            y_prob.extend(yb.cpu().numpy().ravel().tolist())
            y_reg.extend(yr.cpu().numpy().ravel().tolist())
    y_pred = (np.array(y_prob)>=th).astype(int)
    acc = accuracy_score(y_true,y_pred); f1 = f1_score(y_true,y_pred)
    prec= precision_score(y_true,y_pred); rec = recall_score(y_true,y_pred)
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = float("nan")
    mse = mean_squared_error(y_true=np.array(y_true), y_pred=np.array(y_reg))
    mae = mean_absolute_error(np.array(y_true), np.array(y_reg))
    try: pr,_ = pearsonr(np.array(y_true), np.array(y_reg))
    except: pr = float("nan")
    return {"acc":acc,"f1":f1,"precision":prec,"recall":rec,"auc":auc,"mse":mse,"mae":mae,"pearson":pr}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.model_cfg))
    tr = yaml.safe_load(open(args.train_cfg))
    teacher = Teacher(cfg["teacher"]).to(device)

    ds_tr = WindowDataset(args.train_index)
    ds_va = WindowDataset(args.val_index)
    dl_tr = DataLoader(ds_tr, batch_size=tr["train"]["batch_size"], shuffle=True,
                       num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)
    dl_va = DataLoader(ds_va, batch_size=tr["train"]["batch_size"], shuffle=False,
                       num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)

    opt = torch.optim.AdamW(teacher.parameters(), lr=tr["train"]["lr"], weight_decay=tr["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=tr["train"].get("mixed_precision",True))
    best_f1 = -1

    for epoch in range(1, tr["train"]["epochs"]+1):
        teacher.train()
        for b in dl_tr:
            a = b["audio"].to(device)
            v = b["vis"].to(device) if b["vis"] is not None else None
            priv = {k: (b["priv"][k].to(device) if b["priv"][k] is not None else None) for k in b["priv"]}
            yb_t = b["y_bin"].to(device); yr_t = b["y_reg"].to(device)

            with torch.cuda.amp.autocast(enabled=tr["train"].get("mixed_precision",True)):
                yb, yr, _ = teacher(a, v, priv)
                loss, logs = multitask_loss(yb, yb_t, yr, yr_t, tr["loss_weights"]["reg_lambda"])

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), tr["train"]["grad_clip"])
            scaler.step(opt); scaler.update()

        # eval
        m = eval_epoch(teacher, dl_va, device, th=0.5)
        print(f"[Epoch {epoch}] {m}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save({"state_dict": teacher.state_dict(), "metrics": m}, args.ckpt)
            print(f"Saved best teacher → {args.ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--train_cfg", default="configs/training.yaml")
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--val_index", required=True)
    ap.add_argument("--ckpt", default="models/checkpoints/teacher_best.pth")
    args = ap.parse_args()
    main(args)
