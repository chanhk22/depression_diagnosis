import os
import yaml
import glob
import json
import pandas as pd


def mask_clnf_file(txt_in, out_csv, t0):
    """
    CLNF 파일을 처리하여 timestamp가 t0 이상인 데이터만 필터링하여 새로운 CSV 파일로 저장합니다.
    txt_in: CLNF 파일 경로 (텍스트 파일)
    out_csv: 출력 CSV 파일 경로
    t0: 필터링 기준 timestamp (초 단위)
    """
    try:
        # 출력 파일 디렉토리 확인 및 생성
        out_dir = os.path.dirname(out_csv)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)  # 디렉토리가 없으면 생성
        
        with open(txt_in, 'r') as f:
            lines = f.readlines()  # 파일의 모든 라인을 읽어옴
        
        # 첫 번째 줄은 헤더로 간주하고, timestamp를 추출하여 데이터프레임에 추가
        headers = lines[0].strip().split(',')  # 첫 번째 줄을 쉼표로 나누기
        timestamp_index = headers.index(' timestamp')  # 'timestamp' 컬럼의 인덱스 찾기
        
        # 필터링된 데이터를 저장할 리스트
        filtered_data = []

        # 데이터를 한 줄씩 처리
        for line in lines[1:]:  # 첫 번째 줄(헤더)은 제외
            row = line.strip().split(',')  # 쉼표로 나누어 데이터 추출
            timestamp = float(row[timestamp_index])  # timestamp 값

            if timestamp >= float(t0):  # t0 이후의 데이터만 필터링
                filtered_data.append(row)  # 조건에 맞는 데이터만 리스트에 추가
        
        # 필터링된 데이터를 CSV로 저장
        if filtered_data:
            # 파일에 헤더 추가
            with open(out_csv, 'w') as f_out:
                f_out.write(','.join(headers) + '\n')  # 헤더를 첫 번째 줄에 작성
                for row in filtered_data:
                    f_out.write(','.join(row) + '\n')  # 각 행을 쉼표로 구분하여 저장
            print(f"Successfully processed and saved CLNF file: {txt_in}")
        else:
            print(f"No data found after applying the t0 filter for {txt_in}")

    except Exception as e:
        print(f"Error processing CLNF file {txt_in}: {e}")


def mask_covarep_file(csv_in, out_csv, t0):
    # COVAREP 파일은 CSV 파일로, 0.01초마다 1행을 차지한다고 가정
    df = pd.read_csv(csv_in)
    
    # t0를 기반으로 행 번호를 계산 (0.01초마다 1행)
    row_start = int(t0 / 0.01)  # t0를 0.01초로 나누어 몇 번째 행부터 시작할지 계산
    df_filtered = df.iloc[row_start:]  # 해당 행부터 끝까지 데이터 필터링

    # 출력 디렉토리 확인 및 생성
    out_dir = os.path.dirname(out_csv)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  # 디렉토리가 없으면 생성

    df_filtered.to_csv(out_csv, index=False)


def mask_features():
    # config 파일 로드
    with open("configs/default.yaml", encoding='utf-8') as f:
        C = yaml.safe_load(f)

    FEAT_IN   = C['paths']['daic_woz']['features_dir']
    PROC_ROOT = C['outputs']['processed_root']
    TRS_IN    = C['paths']['daic_woz']['transcript_dir']
    
    # t0 값 로드 (audio_trim.py에서 저장한 JSON 파일)
    with open('t0_values.json', 'r') as f:
        t0_values = json.load(f)

    # 처리할 Feature 서브 디렉토리 목록
    for feature_subdir in ["clnf", "covarep"]:
        feature_dir = os.path.join(FEAT_IN, feature_subdir)
        print(f"Processing feature directory: {feature_dir}")
        
        # 각 세션 처리
        for sid in glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv")):
            sid = os.path.basename(sid).split('_')[0]
            print(f"Processing features for session: {sid}")

            # t0 값 가져오기 (없으면 skip)
            if sid not in t0_values:
                print(f"Skipping session {sid} (no t0 value found)")
                continue

            t0 = t0_values[sid]

             # CLNF 텍스트 파일 처리
            if feature_subdir == "clnf":
                for txt_in in glob.glob(os.path.join(feature_dir, f"{sid}_*.txt")): 
                    out_csv = os.path.join(PROC_ROOT, "DAIC-WOZ", "Features", feature_subdir, os.path.basename(txt_in))

                    try:
                        mask_clnf_file(txt_in, out_csv, t0)
                    except Exception as e:
                        print(f"Error processing CLNF file: {txt_in} for session {sid}: {e}")

            # COVAREP CSV 파일 처리
            if feature_subdir == "covarep":
                for csv_in in glob.glob(os.path.join(feature_dir, f"{sid}_*.csv")):
                    out_csv = os.path.join(PROC_ROOT, "DAIC-WOZ", "Features", feature_subdir, os.path.basename(csv_in))

                    try:
                        mask_covarep_file(csv_in, out_csv, t0)
                    except Exception as e:
                        print(f"Error processing COVAREP file: {csv_in} for session {sid}: {e}")

    print("Feature masking t0 filter done.")

if __name__ == "__main__":
    mask_features()
