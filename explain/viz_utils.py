# Top-k feature plot (예: landmark 좌표 TOP5)

# explain/viz_utils.py
import numpy as np
import matplotlib.pyplot as plt
import os

def topk_landmark_bar(shap_values, landmark_names=None, k=10, out_path="topk_landmarks.png"):
    """
    shap_values: 1D array length 136 (68x2) or length 68x2 flattened
    landmark_names: optional list of names per coordinate (e.g., x0,y0,x1,y1,...)
    """
    arr = np.asarray(shap_values).flatten()
    if arr.size % 2 == 0 and arr.size == 136:
        coords = arr
    else:
        coords = arr
    # aggregate per-point by sum of abs(x)+abs(y)
    pts = coords.reshape(-1,2)
    importance = np.abs(pts).sum(axis=1)
    top_idx = np.argsort(importance)[::-1][:k]
    labels = []
    vals = []
    for i in top_idx:
        lab = f"p{i}_x/p{i}_y"
        labels.append(lab)
        vals.append(importance[i])
    plt.figure(figsize=(8,4))
    plt.bar(range(len(vals)), vals, tick_label=labels)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-%d landmark importance (abs x+y)" % k)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return top_idx, vals

def save_shap_heatmap_landmarks(shap_flat, out_png, image_overlay=None, points=None):
    """
    shap_flat: (136,) flattened shap per coordinate.
    points: optionally (68,2) coordinates to overlay.
    This function draws a simple heatmap over points.
    """
    arr = np.asarray(shap_flat).reshape(-1,2)
    importance = np.abs(arr).sum(axis=1)
    # normalize
    imp = (importance - importance.min()) / (importance.ptp() + 1e-9)
    fig, ax = plt.subplots(figsize=(6,6))
    if image_overlay is not None:
        ax.imshow(image_overlay)
    if points is not None:
        xs = points[:,0]; ys = points[:,1]
    else:
        # synthetic grid
        xs = np.linspace(0,1,68)
        ys = np.linspace(0,1,68)
    sc = ax.scatter(xs, ys, c=imp, cmap='Reds', s=60)
    plt.colorbar(sc, ax=ax)
    plt.title("SHAP importance per landmark")
    plt.savefig(out_png)
    plt.close()
    return True
