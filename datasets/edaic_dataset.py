# E-DAIC 로더 (멀티태스크: PHQ score + binary)

# datasets/edaic_dataset.py
import os, pandas as pd
from .base_dataset import BaseWindowDataset, collate_windows
from typing import Callable

def load_labels_edaic(labels_dir):
    """
    labels_dir should contain split CSVs or label CSVs with columns: Participant_ID, gender, PHQ_Binary, PHQ_Score, ...
    returns label_map: dict session-> {'label':int, 'phq_score':float}
    """
    mapping = {}
    # try Detailed_PHQ8_Labels.csv first
    for fname in os.listdir(labels_dir):
        if 'labels' in fname.lower() or 'phq' in fname.lower():
            p = os.path.join(labels_dir, fname)
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            # heuristics to find columns
            if 'Participant_ID' in df.columns and ('PHQ_Binary' in df.columns or 'PHQ_Score' in df.columns):
                for _, r in df.iterrows():
                    sid = str(int(r['Participant_ID'])) if not pd.isna(r['Participant_ID']) else str(r['Participant_ID'])
                    lab = {}
                    if 'PHQ_Binary' in r:
                        lab['label'] = int(r.get('PHQ_Binary', 0))
                    if 'PHQ_Score' in r:
                        lab['phq_score'] = float(r.get('PHQ_Score', 0.0))
                    mapping[sid] = lab
    return mapping

class EdaicDataset(BaseWindowDataset):
    def __init__(self, index_csv, labels_dir, transform=None):
        self.labels = load_labels_edaic(labels_dir)
        super().__init__(index_csv, label_map=self._label_map, transform=transform)

    def _label_map(self, session):
        session = str(session)
        return self.labels.get(session, None)
