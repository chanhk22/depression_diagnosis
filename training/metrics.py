import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    prf_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {"acc":acc, "f1_macro":f1_macro, "f1_weighted":f1_weighted, "precision":prec, "recall":rec, "auc":auc}

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        pearson = pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson = float('nan')
    try:
        spearman = spearmanr(y_true, y_pred)[0]
    except Exception:
        spearman = float('nan')
    return {"mse":mse, "mae":mae, "pearson":pearson, "spearman":spearman}
