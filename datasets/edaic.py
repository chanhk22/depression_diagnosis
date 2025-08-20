# datasets/edaic.py
import os, glob, numpy as np, pandas as pd
from torch.utils.data import Dataset
from features.audio.egemaps_reader import read_ege_llD_from_csv

class EDAICDataset(Dataset):
    """
    학생 입력: audio LLD만 보장(landmark 없음 → zeros).
    교사 입력 특권: OpenFace AUs/pose/gaze 등은 train_kd에서만 별도 로더로 사용.
    """
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        split_csv = os.path.join(cfg["data"]["datasets"]["edaic"]["split_dir"], f"{split}.csv")
        self.split = pd.read_csv(split_csv)  # Participant_ID, gender, PHQ_Binary, PHQ_Score, ...
        self.audio_dir = cfg["data"]["datasets"]["edaic"]["audio_csv_dir"]
        self.ids = self.split["Participant_ID"].astype(str).tolist()

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        # eGeMAPS csv 찾기(파일명 규칙 맞추기)
        cand = glob.glob(os.path.join(self.audio_dir, f"{pid}*_egemaps*.csv"))
        assert len(cand) >= 1, f"eGeMAPS not found for {pid}"
        x_a = read_ege_llD_from_csv(cand[0])           # (T,25)
        x_v = np.zeros((x_a.shape[0], self.cfg["data"]["lmk_dim"]), np.float32)  # no landmarks

        y = float(self.split.loc[self.split["Participant_ID"].astype(str)==pid, "PHQ_Binary"].values[0])
        return {"audio": x_a, "landmark": x_v, "label": y, "meta": {"id": pid, "src": "E-DAIC"}}
