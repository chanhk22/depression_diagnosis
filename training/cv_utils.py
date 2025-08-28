import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def stratified_session_folds(session_metadata_csv, out_dir, n_splits=5, random_state=42):
    """
    session_metadata_csv: CSV with columns session, PHQ_Binary (0/1), Gender
    Writes out CSVs: fold_0_train.csv, fold_0_val.csv etc (each contains session ids)
    """
    df = pd.read_csv(session_metadata_csv)
    if 'PHQ_Binary' not in df.columns:
        raise RuntimeError("session_metadata must contain PHQ_Binary column")
    df['gender_norm'] = df['Gender'].fillna('U').astype(str)
    # create strat key
    df['strat'] = df['PHQ_Binary'].astype(str) + "_" + df['gender_norm']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = df['session'].values
    y = df['strat'].values
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_sessions = df.iloc[train_idx]['session'].tolist()
        val_sessions = df.iloc[val_idx]['session'].tolist()
        pd.DataFrame({"session":train_sessions}).to_csv(f"{out_dir}/fold{fold_idx}_train_sessions.csv", index=False)
        pd.DataFrame({"session":val_sessions}).to_csv(f"{out_dir}/fold{fold_idx}_val_sessions.csv", index=False)
        folds.append((train_sessions, val_sessions))
    return folds
