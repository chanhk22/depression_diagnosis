# datasets/daic.py
from .window_dataset import WindowDataset

def get_daic_windows(index_csv, split="train"):
    return WindowDataset(index_csv=index_csv, split=split, domains=["daic"])
