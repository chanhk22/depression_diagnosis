#!/usr/bin/env bash
set -e

# Test preprocessing.label_mapping
# uniform mapping of session IDs → {PHQ score, binary label, gender, fold}, with automatic corrections, consistency checks, and summary stats
#python -m preprocessing.label_mapping
# Can create window specific dataset insetad of all -> [DAIC-WOZ, E-DAIC, D-VLOG]


python -m precache.window_cache --config configs/default.yaml --dataset all



# 2.splits 생성 (원래 방식으로) 선택 가능 [DAIC-WOZ, E-DAIC, D-VLOG]
python -m training.data_split_manager --dataset D-VLOG

# 3. Combined splits 생성 (DAIC-WOZ + E-DAIC)
python -m training.data_split_manager --create-combined

# 4. 최종 확인
python -m scripts.check_splits

echo "Window cache built."