#!/usr/bin/env bash
# scripts/05_train_student.sh
set -e
CFG=configs/default.yaml
python -u training/train_student.py --config ${CFG} 2>&1 | tee logs/train_student.log
