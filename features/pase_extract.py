# (옵션) PASE+ 임베딩 추출

# features/pase_extract.py
"""
Wrapper for extracting PASE-like embeddings.
This script attempts to use a user-provided PASE model (PASE+). Installing/obtaining PASE+:
- Clone https://github.com/santi-pdp/pase or official PASE+ repo and install
- Place checkpoint file and set PASE_CHECKPOINT env var or pass --pase_ckpt

If PASE is not available, this module falls back to a mel-spectrogram + simple CNN placeholder.

Functions:
- extract_pase_embeddings(wav_paths, out_dir, pase_ckpt=None, device='cpu')
"""
import os, numpy as np, soundfile as sf

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def extract_pase_embeddings(wav_paths, out_dir, pase_ckpt=None, device='cpu'):
    ensure_dir(out_dir)
    try:
        # Try import PASE (user must install PASE+ separately)
        from pase.models.frontend import wf_builder
        import torch
        model = torch.load(pase_ckpt, map_location=device) if pase_ckpt else None
        if model is None:
            raise ImportError("PASE checkpoint not provided.")
        model.eval()
        for wav in wav_paths:
            y, sr = sf.read(wav)
            if y.ndim>1: y = y.mean(axis=1)
            # assume PASE preprocess expects 16k
            # user should ensure sampling rate; here we assume sr==16000 else resample externally
            x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.forward(x)  # model-specific API
            emb_np = emb.cpu().numpy()
            base = os.path.splitext(os.path.basename(wav))[0]
            outp = os.path.join(out_dir, f"{base}_pase.npy")
            np.save(outp, emb_np)
        return True
    except Exception as e:
        # Fallback: mel spectrogram + mean pooling as placeholder
        print("[pase_extract] PASE not available or failed, falling back to mel placeholder:", e)
        import librosa
        import torch
        for wav in wav_paths:
            y, sr = sf.read(wav)
            if y.ndim>1: y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y, sr, 16000)
                sr = 16000
            mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=64, hop_length=160, win_length=400)
            logmel = librosa.power_to_db(mel)
            emb = logmel.mean(axis=1)  # (64,)
            base = os.path.splitext(os.path.basename(wav))[0]
            outp = os.path.join(out_dir, f"{base}_pase_placeholder.npy")
            np.save(outp, emb.astype(np.float32))
        return True
