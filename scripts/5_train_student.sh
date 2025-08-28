#!/usr/bin/env bash
set -e
python -u training/train_student.py \
  --model_cfg configs/model.yaml \
  --train_cfg configs/training.yaml \
  --teacher_ckpt models/checkpoints/teacher_best.pth \
  --train_index data/cache/daic_edaic_train_index.csv \
  --val_index   data/cache/daic_edaic_val_index.csv \
  --ckpt models/checkpoints/student_best.pth
