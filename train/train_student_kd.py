import yaml, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.window_dataset import WindowDataset
from datasets.collate import collate_fn
from models.teacher_student import Teacher, Student
from models.losses import KDLoss, mmd_loss
from utils.seed_utils import set_seed
from utils.metrics import compute_metrics
from tqdm import tqdm

def run_epoch(student, loader, kd=None, teacher=None, mmd_lambda=0.0, optim=None, device="cuda"):
    is_train = optim is not None
    if is_train: student.train()
    else: student.eval()
    if teacher is not None: teacher.eval()

    y_true, y_prob = [], []
    total_loss=0.0
    for batch in tqdm(loader, disable=False):
        lld = torch.tensor(batch["lld"], dtype=torch.float32, device=device)
        y   = torch.tensor(batch["label"], dtype=torch.float32, device=device)
        lmk = batch.get("lmk", None)
        if lmk is not None:
            lmk = torch.tensor(lmk, dtype=torch.float32, device=device)
            # (B,T,136) -> landmark only (no micro) in this MVP
        micro = batch.get("micro", None)
        if micro is not None:
            micro = torch.tensor(micro, dtype=torch.float32, device=device)
        priv = batch.get("priv", None)
        if teacher is not None and priv is not None:
            priv = torch.tensor(priv, dtype=torch.float32, device=device)

        if is_train: optim.zero_grad()
        slogits, sfeat = student(lld, lmk, micro)

        tlogits = None
        if (teacher is not None) and (priv is not None):
            with torch.no_grad():
                tlogits,_ = teacher(lld, priv)

        loss = kd(slogits, tlogits, y) if kd is not None else torch.nn.functional.binary_cross_entropy_with_logits(slogits, y.float())

        # (옵션) 도메인 적응: 간단히 배치 내 도메인 A/B를 두 그룹 뽑아 MMD
        if mmd_lambda>0:
            # 도메인 분리
            domains = batch["domain"]
            dom_a = [i for i,d in enumerate(domains) if d in ["edaic","daic"]]
            dom_b = [i for i,d in enumerate(domains) if d in ["dvlog"]]
            if len(dom_a)>1 and len(dom_b)>1:
                xa = sfeat[dom_a]; xb = sfeat[dom_b]
                loss = loss + mmd_lambda * mmd_loss(xa, xb)

        if is_train:
            loss.backward(); optim.step()
        total_loss += loss.item()*y.size(0)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(torch.sigmoid(slogits).detach().cpu().numpy().tolist())

    avg_loss = total_loss/max(1,len(y_true))
    return avg_loss, compute_metrics(y_true, y_prob)

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmd", type=float, default=0.0)
    args=ap.parse_args()

    set_seed(args.seed)
    env = yaml.safe_load(open(args.env))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_csv = args.index or (Path(env["paths"]["cache_root"]) / env["outputs"]["index_csv"])

    # 데이터셋: 세 도메인 모두 포함
    train_set = WindowDataset(index_csv, split="train", domains=["edaic","daic","dvlog"])
    dev_set   = WindowDataset(index_csv, split="dev",   domains=["edaic","daic","dvlog"])
    test_sets = {d: WindowDataset(index_csv, split="test", domains=[d]) for d in ["edaic","daic","dvlog"]}

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_set,   batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    # 모델
    student = Student(d=256, use_visual=True).to(device)

    # Teacher 로드(있을 때만 KD)
    teacher = Teacher(d=256, priv_in=None).to(device)  # priv_in 나중에 덮어씀
    ckpt = Path(env["paths"]["out_root"]) / env["train"]["ckpt_dir"] / "teacher_best.pt"
    kd = None
    if ckpt.exists():
        # priv_in 추정을 위해 dummy batch
        sample = next(iter(train_loader))
        if "priv" in sample:
            pin = sample["priv"].shape[-1]
            teacher = Teacher(d=256, priv_in=pin).to(device)
            teacher.load_state_dict(torch.load(ckpt, map_location=device))
            kd = KDLoss(T=2.0, alpha=0.5)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)

    best_auc = -1
    ckpt_dir = Path(env["paths"]["out_root"]) / env["train"]["ckpt_dir"]; ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs+1):
        tr_loss, tr_met = run_epoch(student, train_loader, kd, teacher, args.mmd, optim, device)
        dv_loss, dv_met = run_epoch(student, dev_loader, kd, teacher, args.mmd, None, device)
        print(f"[Ep{ep}] Train AUC {tr_met['AUROC']:.3f} | Dev AUC {dv_met['AUROC']:.3f}")
        if dv_met["AUROC"] > best_auc:
            best_auc = dv_met["AUROC"]
            torch.save(student.state_dict(), ckpt_dir / "student_best.pt")
            print("Saved best student.")

    # 각 도메인 test
    student.load_state_dict(torch.load(ckpt_dir / "student_best.pt", map_location=device))
    for name, ds in test_sets.items():
        loader = DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
        _, mt = run_epoch(student, loader, kd, teacher, args.mmd, None, device)
        print(f"[TEST:{name}] {mt}")
