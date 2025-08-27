# DAIC-WOZ 로더 (CLNF + COVAREP)

# datasets/daic_dataset.py
import os, pandas as pd
from .base_dataset import BaseWindowDataset

def load_daic_labels(labels_dir):
    # DAIC label files might have Participant_ID, PHQ_Score, PHQ_Binary
    mapping = {}
    for fname in os.listdir(labels_dir):
        if fname.lower().endswith('.csv'):
            df = pd.read_csv(os.path.join(labels_dir,fname))
            if 'Participant_ID' in df.columns:
                for _, r in df.iterrows():
                    sid = str(int(r['Participant_ID'])) if not pd.isna(r['Participant_ID']) else str(r['Participant_ID'])
                    mapping[sid] = {}
                    if 'PHQ_Binary' in r:
                        mapping[sid]['label'] = int(r['PHQ_Binary'])
                    if 'PHQ_Score' in r:
                        mapping[sid]['phq_score'] = float(r['PHQ_Score'])
    return mapping

class DaicDataset(BaseWindowDataset):
    def __init__(self, index_csv, labels_dir, transform=None):
        self.labels = load_daic_labels(labels_dir)
        super().__init__(index_csv, label_map=self._label_map, transform=transform)

    def _label_map(self, session):
        session = str(session)
        return self.labels.get(session, None)
