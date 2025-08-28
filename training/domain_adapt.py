# GRL/MMD 기반 D-VLOG adaptation

import os, yaml, argparse, torch
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.base_dataset import default_collate_fn
from models.student import Student
from training.utils_losses import multitask_loss, domain_losses
from train_teacher import eval_epoch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.model_cfg))
    tr  = yaml.safe_load(open(args.train_cfg))

    model = Student(cfg["student"]).to(device)
    sd = torch.load(args.student_ckpt, map_location=device)
    model.load_state_dict(sd["state_dict"])

    ds_src = WindowDataset(args.src_index)   # labeled (DAIC/E-DAIC)
    ds_tgt = WindowDataset(args.tgt_index)   # unlabeled (D-VLOG)
    dl_src = DataLoader(ds_src, batch_size=tr["train"]["batch_size"], shuffle=True,
                        num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)
    dl_tgt = DataLoader(ds_tgt, batch_size=tr["train"]["batch_size"], shuffle=True,
                        num_workers=tr["train"]["num_workers"], collate_fn=default_collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["train"]["lr"], weight_decay=tr["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=tr["train"].get("mixed_precision",True))

    for epoch in range(1, tr["train"]["epochs"]+1):
        model.train()
        for (bs, bt) in zip(dl_src, dl_tgt):
            a_s = bs["audio"].to(device)
            v_s = bs["vis"].to(device) if (bs["vis"] is not None and getattr(model,"use_landmarks",False)) else None
            yb_s = bs["y_bin"].to(device); yr_s = bs["y_reg"].to(device)

            a_t = bt["audio"].to(device)
            v_t = bt["vis"].to(device) if (bt["vis"] is not None and getattr(model,"use_landmarks",False)) else None

            with torch.cuda.amp.autocast(enabled=tr["train"].get("mixed_precision",True)):
                yb, yr, h_s = model(a_s, v_s)
                _,  _, h_t = model(a_t, v_t)
                l_mt, _ = multitask_loss(yb, yb_s, yr, yr_s, tr["loss_weights"]["reg_lambda"])
                l_da, _ = domain_losses(h_s.mean(0, keepdim=True), h_t.mean(0, keepdim=True),
                                        grl_lambda=tr["loss_weights"]["grl_lambda"],
                                        mmd_lambda=tr["loss_weights"]["mmd_lambda"],
                                        use_grl=tr["da"]["use_grl"], use_mmd=tr["da"]["use_mmd"])
                loss = l_mt + l_da

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["train"]["grad_clip"])
            scaler.step(opt); scaler.update()

        print(f"[DA Epoch {epoch}] done")

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.ckpt)
    print(f"Saved domain-adapted student → {args.ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--train_cfg", default="configs/training.yaml")
    ap.add_argument("--student_ckpt", default="models/checkpoints/student_best.pth")
    ap.add_argument("--src_index", required=True)
    ap.add_argument("--tgt_index", required=True)
    ap.add_argument("--ckpt", default="models/checkpoints/student_adapted.pth")
    args = ap.parse_args()
    main(args)
