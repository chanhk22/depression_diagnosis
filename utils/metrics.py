import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss

def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob>=thr).astype(int)
    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan")
    out["AUPRC"] = average_precision_score(y_true, y_prob)
    out["F1"]    = f1_score(y_true, y_pred)
    out["MCC"]   = matthews_corrcoef(y_true, y_pred)
    out["ECE"]   = ece_score(y_true, y_prob, n_bins=15)
    out["Brier"] = brier_score_loss(y_true, y_prob)
    return out

def ece_score(y_true, y_prob, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    ece=0.0; N=len(y_true)
    for i in range(n_bins):
        mask = (y_prob>bins[i]) & (y_prob<=bins[i+1])
        if mask.sum()==0: continue
        conf = y_prob[mask].mean()
        acc = (y_true[mask]==(y_prob[mask]>=0.5)).mean()
        ece += (mask.sum()/N)*abs(acc-conf)
    return ece
