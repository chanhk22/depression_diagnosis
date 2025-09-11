import os
import yaml
import glob
import json
from preprocessing.mask_csv import mask_by_t0  # mask_by_t0 함수 임포트
import traceback
import logging
import shutil

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


with open("configs/default.yaml", encoding='utf-8') as f:
    C = yaml.safe_load(f)

FEAT_IN   = C['paths']['e_daic']['features_dir']
PROC_ROOT = C['outputs']['processed_root']

# t0 값 로드 (t0.json 파일)
with open('t0_values.json', 'r') as f:
    t0_values = json.load(f)

# 필요한 디렉토리 생성
os.makedirs(os.path.join(PROC_ROOT, "E-DAIC/Features/densenet201"), exist_ok=True)
os.makedirs(os.path.join(PROC_ROOT, "E-DAIC/Features/mfcc"), exist_ok=True)
os.makedirs(os.path.join(PROC_ROOT, "E-DAIC/Features/openface_pose_gaze_au"), exist_ok=True)
os.makedirs(os.path.join(PROC_ROOT, "E-DAIC/Features/vgg16"), exist_ok=True)

def process_files():
    # 각 feature 디렉토리 순회
    for feature_subdir in ["densenet201", "mfcc", "openface_pose_gaze_au", "vgg16"]:
        feature_dir = os.path.join(FEAT_IN, feature_subdir)
        out_dir = os.path.join(PROC_ROOT, "E-DAIC", "Features", feature_subdir)
        os.makedirs(out_dir, exist_ok=True)

        logging.info(f"Processing feature directory: {feature_dir}")

        # 해당 feature 디렉토리 안의 모든 csv 파일 처리
        csv_files = glob.glob(os.path.join(feature_dir, "*.csv"))

        for csv_in in csv_files:
            filename = os.path.basename(csv_in)
            sid = filename.split("_")[0]   # 예: "490_P_xxx.csv" -> "490"

            #출력 파일명을 feature_subdir 이름 그대로
            out_filename = f"{sid}_{feature_subdir}.csv"
            out_csv = os.path.join(out_dir, filename)

            # 이미 처리된 파일 건너뜀
            if os.path.exists(out_csv):
                logging.info(f"File {out_csv} already exists. Skipping.")
                continue

            # t0 값 확인
            t0 = t0_values.get(sid, None)

            try:
                if t0 is None:
                    # 원본 그대로 복사
                    shutil.copy(csv_in, out_csv)
                    logging.info(f"[COPY] {filename} -> {out_csv}")
                else:
                    # t0 마스킹 적용
                    mask_by_t0(csv_in, out_csv, t0)
                    logging.info(f"[MASK] {filename} with t0={t0} -> {out_csv}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

    logging.info("DAIC head cut + CSV t0 filter done.")


# Main execution
if __name__ == "__main__":
    try:
        process_files()
    except Exception as e:
        logging.error(f"An error occurred during file processing: {e}")
        traceback.print_exc()
