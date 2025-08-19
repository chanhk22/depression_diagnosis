import yaml, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.collate import collate_fn
from models.teacher_student import Student
from utils.metrics import compute_metrics

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--domains", nargs="+", default=["edaic","daic","dvlog"])
    ap.add_argument("--bs", type=int, default=64)
    args=ap.parse_args()

    env = yaml.safe_load(open(args.env))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_csv = args.index or (Path(env["paths"]["cache_root"]) / env["outputs"]["index_csv"])
    ds = WindowDataset(index_csv, split=args.split, domains=args.domains)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    model = Student(d=256, use_visual=True).to(device)
    if args.ckpt is None:
        args.ckpt = Path(env["paths"]["out_root"]) / env["train"]["ckpt_dir"] / "student_best.pt"
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            lld = torch.tensor(batch["lld"], dtype=torch.float32, device=device)
            lmk = torch.tensor(batch["lmk"], dtype=torch.float32, device=device) if "lmk" in batch else None
            micro = torch.tensor(batch["micro"], dtype=torch.float32, device=device) if "micro" in batch else None
            logits,_ = model(lld, lmk, micro)
            y_true.extend(batch["label"].numpy().tolist())
            y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    print(compute_metrics(np.array(y_true), np.array(y_prob)))
