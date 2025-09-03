# preprocess_landmarks.py
import os
import glob
import yaml
import numpy as np

from preprocessing.clnf_parser import read_clnf_features_txt, normalize_landmarks
from preprocessing.dvlog_visual_parser import normalize_dvlog_visual


class LandmarkPreprocessor:
    def __init__(self, config_path="configs/default.yaml"):
        with open(config_path, encoding='utf-8') as f:
            C = yaml.safe_load(f)
        self.DVLOG_RAW = C['paths']['dvlog']['visual']
        self.DVLOG_PROCESSED = C['processed']['dvlog']['visual']
        self.DAIC_PROCESSED = C['processed']['daic_woz']['features']

    def process_daic_clnf(self, input_dir=None, output_dir=None, method="interocular"):
        """
        Parse DAIC-WOZ CLNF features:
          - Read raw txt/CSV (x0..x67, y0..y67 → (T,68,2))
          - Normalize (interocular/zscore/none)
          - Save to npy
        """
        input_dir = input_dir or os.path.join(self.DAIC_PROCESSED, "clnf")
        output_dir = output_dir or os.path.join(self.DAIC_PROCESSED, "clnf")
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(input_dir):
            if not fname.endswith(".txt"):
                continue

            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.replace(".txt", ".npy"))

            # 1. parse raw CLNF txt → (T,68,2)
            _, pts = read_clnf_features_txt(in_path)

            # 2. normalize
            pts_norm = normalize_landmarks(pts, method=method)

            # 3. save
            np.save(out_path, pts_norm.astype(np.float32))
            print(f"[DAIC CLNF] {fname}: {pts.shape} -> saved {out_path}")

            # 4. remove original .txt
            try:
                os.remove(in_path)
                print(f"  removed original {in_path}")
            except Exception as e:
                print(f"  failed to remove {in_path}: {e}")

    def process_dvlog(self, method="interocular"):
        """
        Parse and normalize D-VLOG visual.npy:
          - Load pre-extracted (T,68,2) landmarks
          - Normalize (interocular/zscore/none)
          - Save to npy
        """
        for f in glob.glob(os.path.join(self.DVLOG_RAW, "*_visual.npy")):
            pts_norm = normalize_dvlog_visual(f, method=method)
            out = f.replace(self.DVLOG_RAW, self.DVLOG_PROCESSED).replace("visual.npy", "landmarks.npy")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, pts_norm)
            print("[DVLOG] saved", out)

    def run_all(self):
        self.process_daic_clnf()
        self.process_dvlog()


if __name__ == "__main__":
    preprocessor = LandmarkPreprocessor()
    preprocessor.run_all()
