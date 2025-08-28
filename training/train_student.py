import os, yaml, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.base_dataset import default_collate_fn
from models.student import Student
from models.teacher import Teacher
from training.utils_losses import multitask_loss, kd_loss
from train_teacher import eval_epoch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.model_cfg))
    tr  = yaml.safe_load(open(args.train_cfg))

    student = Student(cfg["student"]).to(device)
    teacher = Teacher(cfg["teacher"]).to(device).eval()
    td = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(td["state_dict"])
    for p in teacher.parameters(): p.requires_grad=False

    ds_tr = WindowDataset(args.train_index)
    ds_va = WindowDataset(args.val_index)
    dl_tr = DataLoader(ds_tr, batch_size=tr["train"]["batch_size"], shuffle=True,
                       num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)
    dl_va = DataLoader(ds_va, batch_size=tr["train"]["batch_size"], shuffle=False,
                       num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)

    opt = torch.optim.AdamW(student.parameters(), lr=tr["train"]["lr"], weight_decay=tr["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=tr["train"].get("mixed_precision",True))
    best_f1 = -1

    for epoch in range(1, tr["train"]["epochs"]+1):
        student.train()
        for b in dl_tr:
            a = b["audio"].to(device)
            v = b["vis"].to(device) if (b["vis"] is not None and getattr(student,"use_landmarks",False)) else None
            yb_t = b["y_bin"].to(device)
            yr_t = b["y_reg"].to(device)

            with torch.cuda.amp.autocast(enabled=tr["train"].get("mixed_precision",True)):
                # teacher signals
                with torch.no_grad():
                    yb_T, yr_T, h_T = teacher(a, v=None, priv=None)   # teacher uses its own modalities in pretraining; here distill logits
                yb_S, yr_S, h_S = student(a, v)
                l_mt, logs = multitask_loss(yb_S, yb_t, yr_S, yr_t, tr["loss_weights"]["reg_lambda"])
                l_kd = kd_loss(yb_S, yb_T, tr["kd"]["T"])
                loss = l_mt + tr["loss_weights"]["kd_lambda"] * l_kd

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), tr["train"]["grad_clip"])
            scaler.step(opt); scaler.update()

        m = eval_epoch(student, dl_va, device, th=0.5)
        print(f"[Student E{epoch}] {m}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save({"state_dict": student.state_dict(), "metrics": m}, args.ckpt)
            print(f"Saved best student → {args.ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--train_cfg", default="configs/training.yaml")
    ap.add_argument("--teacher_ckpt", default="models/checkpoints/teacher_best.pth")
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--val_index", required=True)
    ap.add_argument("--ckpt", default="models/checkpoints/student_best.pth")
    args = ap.parse_args()
    main(args)
