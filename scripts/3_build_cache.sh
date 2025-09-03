#!/usr/bin/env bash
set -e
python -m precache.window_cache --config configs/default.yaml --dataset all

# 분할 생성 (훨씬 빨라짐)
python training.data_split_manager

echo "Window cache built."