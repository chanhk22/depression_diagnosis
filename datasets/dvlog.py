# datasets/dvlog.py
from .window_dataset import WindowDataset

def get_dvlog_windows(index_csv, split="train"):
    return WindowDataset(index_csv=index_csv, split=split, domains=["dvlog"])
