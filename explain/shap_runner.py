# explain/shap_runner.py
import shap, numpy as np, torch
from models.student import StudentModel

def model_predict_flat(X_flat, model, device, T, D):
    X = X_flat.reshape(-1, T, D)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits,_,_ = model(X_t, torch.zeros((X_t.shape[0], T, 136)))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def run_shap(model_ckpt, sample_npz_list, cfg):
    device = torch.device(cfg['training']['device'])
    model = StudentModel().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    T = int(cfg['window']['length_s'] * cfg['features']['target_hz'])
    D = cfg['features']['lld_dim']
    X=[]
    for npz in sample_npz_list:
        data = np.load(npz)
        a = data['audio']
        if a.shape[0] != T: import numpy as npr; a = npr.resize(a, (T, D))
        X.append(a.reshape(-1))
    X = np.stack(X)
    explainer = shap.KernelExplainer(lambda z: model_predict_flat(z, model, device, T, D), X[:50])
    shap_values = explainer.shap_values(X[50:60], nsamples=200)
    return shap_values
