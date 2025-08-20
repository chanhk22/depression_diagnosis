# datasets/daic.py
import os, glob, numpy as np
from torch.utils.data import Dataset
from features.visual.landmark_reader import parse_clnf_features_txt
from features.audio.egemaps_reader import read_ege_llD_from_csv

class DAICDataset(Dataset):
    def __init__(self, cfg, split="train", use_wav_ege=True):
        self.cfg = cfg
        self.split = split
        self.wav_dir = cfg["data"]["datasets"]["daic"]["wav_dir"]
        self.clnf_dir = cfg["data"]["datasets"]["daic"]["clnf_dir"]
        self.use_wav_ege = use_wav_ege
        # split 파일은 E-DAIC label csv를 재활용하거나, DAIC 전용 split csv를 별도로 둔다고 가정
        # 여기서는 간단히 모든 세션을 glob으로 로드(실전은 split csv 권장)
        self.ids = sorted(set([os.path.basename(p).split('_')[0] for p in glob.glob(os.path.join(self.clnf_dir, "*_CLNF_features.txt"))]))

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        # audio LLD: wav에서 추출했거나 미리 csv로 저장해둔 경로를 로드
        egemaps_csv = os.path.join(self.wav_dir, f"{pid}_OpenSMILE_eGeMAPS.csv")
        if os.path.isfile(egemaps_csv):
            x_a = read_ege_llD_from_csv(egemaps_csv)
        else:
            # 전처리에서 미리 만들어두는 것을 권장. 여기선 빈 틀 방지:
            x_a = np.zeros((1, self.cfg["data"]["lld_dim"]), np.float32)

        # landmarks
        clnf_txt = os.path.join(self.clnf_dir, f"{pid}_CLNF_features.txt")
        x_v = parse_clnf_features_txt(clnf_txt)

        # (옵션) Ellie 마스킹은 preprocessing 단계에서 프레임 제거 권장
        y = 0.0  # 라벨 로딩은 너의 split 파일 형식에 맞춰 채우기
        return {"audio": x_a, "landmark": x_v, "label": y, "meta": {"id": pid, "src": "DAIC"}}
