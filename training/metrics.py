# training/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr


def classification_metrics(y_true, y_prob, threshold=0.5, average='macro'):
    """
    Compute classification metrics
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Decision threshold
        average: Averaging method for multi-class metrics
    
    Returns:
        dict: Dictionary of metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # Basic metrics
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0, average='binary')
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0, average='binary')
    
    # Multi-class metrics if needed
    if len(np.unique(y_true)) > 2 or average != 'binary':
        metrics['f1_macro'] = f1_score(y_true, y_pred, zero_division=0, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, zero_division=0, average='weighted')
        
        # Precision/Recall for each class
        prec_rec_f1 = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average=None)
        for i, (p, r, f) in enumerate(zip(prec_rec_f1[0], prec_rec_f1[1], prec_rec_f1[2])):
            metrics[f'precision_class_{i}'] = p
            metrics[f'recall_class_{i}'] = r
            metrics[f'f1_class_{i}'] = f
    
    # AUC (handle edge cases)
    try:
        if len(np.unique(y_true)) > 1:  # Need at least 2 classes
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc'] = float('nan')
    except Exception:
        metrics['auc'] = float('nan')
    
    # Confusion matrix components
    if len(np.unique(y_true)) == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp) 
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            
            # Additional derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # same as recall
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # negative predictive value
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # positive predictive value (same as precision)
            
        except ValueError:
            # Handle case where confusion matrix cannot be computed
            pass
    
    return metrics


def regression_metrics(y_true, y_pred):
    """
    Compute regression metrics
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
    
    Returns:
        dict: Dictionary of metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'pearson': float('nan'),
            'spearman': float('nan')
        }
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation metrics
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson'] = pearson_corr
            metrics['pearson_p'] = pearson_p
        else:
            metrics['pearson'] = float('nan')
            metrics['pearson_p'] = float('nan')
    except Exception:
        metrics['pearson'] = float('nan')
        metrics['pearson_p'] = float('nan')
    
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman'] = spearman_corr
            metrics['spearman_p'] = spearman_p
        else:
            metrics['spearman'] = float('nan')
            metrics['spearman_p'] = float('nan')
    except Exception:
        metrics['spearman'] = float('nan')
        metrics['spearman_p'] = float('nan')
    
    # R-squared
    try:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        metrics['r2'] = float('nan')
    
    return metrics


def compute_all_metrics(y_true_bin, y_prob_bin, y_true_reg, y_pred_reg, threshold=0.5):
    """
    Compute both classification and regression metrics
    
    Args:
        y_true_bin: True binary labels
        y_prob_bin: Predicted probabilities
        y_true_reg: True continuous values  
        y_pred_reg: Predicted continuous values
        threshold: Decision threshold for binary classification
    
    Returns:
        dict: Combined metrics dictionary
    """
    clf_metrics = classification_metrics(y_true_bin, y_prob_bin, threshold)
    reg_metrics = regression_metrics(y_true_reg, y_pred_reg)
    
    # Combine with prefixes to avoid naming conflicts
    combined = {}
    for k, v in clf_metrics.items():
        combined[f'clf_{k}'] = v
    for k, v in reg_metrics.items():
        combined[f'reg_{k}'] = v
        
    return combined


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics"""
    print(f"\n=== {title} ===")
    
    # Group metrics by type
    clf_metrics = {k.replace('clf_', ''): v for k, v in metrics.items() if k.startswith('clf_')}
    reg_metrics = {k.replace('reg_', ''): v for k, v in metrics.items() if k.startswith('reg_')}
    other_metrics = {k: v for k, v in metrics.items() if not k.startswith(('clf_', 'reg_'))}
    
    # Print classification metrics
    if clf_metrics:
        print("\nClassification:")
        key_metrics = ['acc', 'f1', 'precision', 'recall', 'auc']
        for key in key_metrics:
            if key in clf_metrics:
                print(f"  {key:12}: {clf_metrics[key]:.4f}")
        
        # Print confusion matrix if available
        if all(k in clf_metrics for k in ['tp', 'tn', 'fp', 'fn']):
            print(f"\nConfusion Matrix:")
            print(f"  TP: {clf_metrics['tp']:4d}  FN: {clf_metrics['fn']:4d}")
            print(f"  FP: {clf_metrics['fp']:4d}  TN: {clf_metrics['tn']:4d}")
    
    # Print regression metrics
    if reg_metrics:
        print("\nRegression:")
        key_metrics = ['mse', 'mae', 'rmse', 'pearson', 'r2']
        for key in key_metrics:
            if key in reg_metrics:
                print(f"  {key:12}: {reg_metrics[key]:.4f}")
    
    # Print other metrics
    if other_metrics:
        print("\nOther:")
        for key, value in other_metrics.items():
            print(f"  {key:12}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key:12}: {value}")