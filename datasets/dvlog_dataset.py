# D-VLOG 로더 (acoustic.npy + visual.npy)

# datasets/dvlog_dataset.py
import os, numpy as np, pandas as pd
from .base_dataset import BaseWindowDataset

def load_dvlog_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    mapping = {}
    # expect columns: id or filename, label (depression/normal or 0/1), duration, fold
    for _, r in df.iterrows():
        sid = str(r.get('id', r.get('participant', r.get('filename', r.get('session')))))
        lab = {}
        # try multiple column names
        if 'label' in r.index:
            v = r['label']
            # convert string labels
            if isinstance(v, str):
                lab['label'] = 1 if v.lower().startswith('depress') else 0
            else:
                lab['label'] = int(v)
        for c in ['PHQ_Binary','PHQ_Score','duration','gender','fold']:
            if c in r.index:
                lab[c] = r[c]
        mapping[sid] = lab
    return mapping

class DvlogDataset(BaseWindowDataset):
    def __init__(self, index_csv, dvlog_labels_csv=None, transform=None):
        self.dv_labels = load_dvlog_labels(dvlog_labels_csv) if dvlog_labels_csv else {}
        super().__init__(index_csv, label_map=self._label_map, transform=transform)

    def _label_map(self, session):
        session = str(session)
        return self.dv_labels.get(session, None)
