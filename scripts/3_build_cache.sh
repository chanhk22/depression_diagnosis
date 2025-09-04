#!/usr/bin/env bash
set -e

# Test preprocessing.label_mapping
# uniform mapping of session IDs → {PHQ score, binary label, gender, fold}, with automatic corrections, consistency checks, and summary stats
python -m preprocessing.label_mapping



python -m precache.window_cache --config configs/default.yaml --dataset all

# 분할 생성 (훨씬 빨라짐)
#python training.data_split_manager

echo "Window cache built."