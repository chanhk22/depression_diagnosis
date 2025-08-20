# scripts/train_student_kd.sh
export DATA_RAW_ROOT=/abs/path/to/data_raw
python -m train.train_kd --config configs/default.yaml
