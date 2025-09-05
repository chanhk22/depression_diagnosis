# debug_label_keys.py
import yaml
from preprocessing.label_mapping import load_labels
import os

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 라벨 로딩
print("=== Loading Labels with Debug ===")
labels_dir = os.path.dirname(cfg['labels']['daic_woz']['train_split'])
labels = load_labels(labels_dir, dataset_hint='DAIC-WOZ')

print(f"\nTotal labels loaded: {len(labels)}")

# 300, 301, 302 키 확인
test_sessions = ['300', '301', '302']

print(f"\n=== Key Analysis ===")
for session in test_sessions:
    print(f"\nSession {session}:")
    
    # 다양한 형태로 확인
    forms = [
        session,           # '300'
        int(session),      # 300
        f"{int(session):03d}",  # '300' (padded)
    ]
    
    for form in forms:
        exists = form in labels
        print(f"  '{form}' (type: {type(form).__name__}): {exists}")
        if exists:
            print(f"    Label data: {labels[form]}")

# 실제 키들의 타입과 값 확인
print(f"\n=== Actual Keys (first 10) ===")
for i, key in enumerate(list(labels.keys())[:10]):
    print(f"  Key {i}: '{key}' (type: {type(key).__name__})")

# 300 근처 키들 찾기
print(f"\n=== Keys containing '300' ===")
for key in labels.keys():
    if '300' in str(key):
        print(f"  Key: '{key}' (type: {type(key).__name__})")