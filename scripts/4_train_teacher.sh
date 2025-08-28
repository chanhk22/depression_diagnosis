#!/usr/bin/env bash
set -e
python -u training/train_teacher.py \
  --model_cfg configs/model.yaml \
  --train_cfg configs/training.yaml \
  --train_index data/cache/daic_edaic_train_index.csv \
  --val_index   data/cache/daic_edaic_val_index.csv \
  --ckpt models/checkpoints/teacher_best.pth
