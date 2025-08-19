# datasets/edaic.py
from .window_dataset import WindowDataset

def get_edaic_windows(index_csv, split="train"):
    # domains filter "edaic"
    return WindowDataset(index_csv=index_csv, split=split, domains=["edaic"])
