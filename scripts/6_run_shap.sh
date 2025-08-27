#!/usr/bin/env bash
# scripts/06_run_shap.sh
set -e
CFG=configs/default.yaml
python -u explain/run_shap.py --config ${CFG} 2>&1 | tee logs/run_shap.log
