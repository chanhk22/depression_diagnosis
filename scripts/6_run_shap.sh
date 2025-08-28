#!/usr/bin/env bash
set -e
python -u explain/shap_runner.py \
  --model_ckpt models/checkpoints/student_adapted.pth \
  --index_csv data/cache/daic_edaic_val_index.csv \
  --out_dir data/shap
