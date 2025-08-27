#!/usr/bin/env bash
# scripts/04_train_teacher.sh
set -e
CFG=configs/default.yaml
python -u training/train_teacher.py --config ${CFG} 2>&1 | tee logs/train_teacher.log
