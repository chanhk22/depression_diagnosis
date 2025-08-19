import yaml, torch, shap, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.collate import collate_fn
from models.teacher_student import Student

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--domains", nargs="+", default=["dvlog"])
    ap.add_argument("--background", type=int, default=32)
    ap.add_argument("--samples", type=int, default=8)
    args=ap.parse_args()

    env = yaml.safe_load(open(args.env))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_csv = args.index or (Path(env["paths"]["cache_root"]) / env["outputs"]["index_csv"])

    ds = WindowDataset(index_csv, split=args.split, domains=args.domains)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = Student(d=256, use_visual=True).to(device)
    ckpt = args.ckpt or (Path(env["paths"]["out_root"]) / env["train"]["ckpt_dir"] / "student_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # background 집합
    bg_lld, bg_lmk, bg_micro = [], [], []
    for i, batch in enumerate(loader):
        if i>=args.background: break
        bg_lld.append(torch.tensor(batch["lld"], dtype=torch.float32, device=device))
        if "lmk" in batch:   bg_lmk.append(torch.tensor(batch["lmk"], dtype=torch.float32, device=device))
        if "micro" in batch: bg_micro.append(torch.tensor(batch["micro"], dtype=torch.float32, device=device))
    bg_lld = torch.cat(bg_lld,0)
    bg_lmk = torch.cat(bg_lmk,0) if len(bg_lmk)>0 else None
    bg_micro = torch.cat(bg_micro,0) if len(bg_micro)>0 else None

    # 모델 래퍼: lld(+lmk/micro) -> prob
    def fwd(inputs):
        lld, lmk, micro = inputs
        logits,_ = model(lld, lmk, micro)
        return torch.sigmoid(logits)

    # GradientExplainer (속도 고려)
    explainer = shap.GradientExplainer(
        (model, lambda lld,lmk,micro: model(lld,lmk,micro)[0]),
        (bg_lld, bg_lmk, bg_micro)
    )

    # 일부 샘플에 대해 shap 값
    X_lld, X_lmk, X_micro = [], [], []
    for i, batch in enumerate(loader):
        if i>=args.samples: break
        X_lld.append(torch.tensor(batch["lld"], dtype=torch.float32, device=device))
        X_lmk.append(torch.tensor(batch["lmk"], dtype=torch.float32, device=device) if "lmk" in batch else None)
        X_micro.append(torch.tensor(batch["micro"], dtype=torch.float32, device=device) if "micro" in batch else None)

    X_lld = torch.cat([x for x in X_lld],0)
    X_lmk = torch.cat([x for x in X_lmk if x is not None],0) if any(x is not None for x in X_lmk) else None
    X_micro = torch.cat([x for x in X_micro if x is not None],0) if any(x is not None for x in X_micro) else None

    shap_vals = explainer.shap_values((X_lld, X_lmk, X_micro))
    print("Computed SHAP values.")
