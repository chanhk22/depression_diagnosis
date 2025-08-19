import numpy as np, pandas as pd, yaml, glob, os
from pathlib import Path

def resample_to_T(arr, T):
    t = arr.shape[0]
    if t == T: return arr.astype(np.float32)
    x0 = np.linspace(0,1,t); x1 = np.linspace(0,1,T)
    return np.stack([np.interp(x1, x0, arr[:,i]) for i in range(arr.shape[1])], axis=1).astype(np.float32)

def make_windows(arr, win, hop):
    out = []
    T = arr.shape[0]
    for s in range(0, max(1, T - win + 1), hop):
        out.append(arr[s:s+win])
    return np.stack(out, 0) if out else np.zeros((0,win,arr.shape[1]), dtype=np.float32)

def zscore(arr, mean=None, std=None):
    if mean is None: mean = arr.mean(axis=(0,1), keepdims=True)
    if std is None:  std  = arr.std(axis=(0,1), keepdims=True) + 1e-8
    return (arr-mean)/std, mean, std

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env.yaml")
    ap.add_argument("--win", default="configs/window.yaml")
    args = ap.parse_args()
    env = yaml.safe_load(open(args.env))
    wcfg = yaml.safe_load(open(args.win))

    cache = Path(env["paths"]["cache_root"])
    audio_dir = cache / env["outputs"]["audio_lld_dir"]
    lmk_dir   = cache / env["outputs"]["lmk_dir"]
    micro_dir = cache / env["outputs"]["micro_dir"]
    priv_dir  = cache / env["outputs"]["priv_dir"]
    mean_shape_path = cache / env["outputs"]["mean_shape"]
    if mean_shape_path.exists():
        mean_shape = np.load(mean_shape_path)
    else:
        mean_shape = None

    out_dir = cache / env["outputs"]["windows_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    # 스플릿/라벨 로딩
    # E-DAIC
    ed_root = Path(env["paths"]["edaic_root"])
    def load_split(path): 
        df = pd.read_csv(ed_root / path)
        assert "Participant_ID" in df.columns and "PHQ_Binary" in df.columns
        return df[["Participant_ID","PHQ_Binary"]].rename(columns={"Participant_ID":"pid","PHQ_Binary":"label"})
    ed_train = load_split(env["edaic"]["split_train_csv"]); ed_train["split"]="train"; ed_train["domain"]="edaic"
    ed_dev   = load_split(env["edaic"]["split_dev_csv"]);   ed_dev["split"]="dev";   ed_dev["domain"]="edaic"
    ed_test  = load_split(env["edaic"]["split_test_csv"]);  ed_test["split"]="test";  ed_test["domain"]="edaic"
    ed_all = pd.concat([ed_train,ed_dev,ed_test],0)

    # DAIC
    da_root = Path(env["paths"]["daic_root"])
    da_labels = pd.read_csv(da_root / env["daic"]["label_csv"])  # cols: pid, split, PHQ_Binary
    da_labels = da_labels.rename(columns={"PHQ_Binary":"label"})
    da_labels["domain"]="daic"

    # D-VLOG
    dv_root = Path(env["paths"]["dvlog_root"])
    dv_meta = pd.read_csv(dv_root / env["dvlog"]["meta_csv"])    # cols: id,label,fold
    dv_meta = dv_meta.rename(columns={"id":"pid","fold":"split"})
    dv_meta["domain"]="dvlog"

    meta_all = pd.concat([ed_all, da_labels[["pid","label","split","domain"]], dv_meta[["pid","label","split","domain"]]], 0)

    # 파라미터
    audio_hz = int(wcfg["audio_hz"])
    win = int(wcfg["window_sec"]*audio_hz)
    hop = int(wcfg["hop_sec"]*audio_hz)

    # 스캔 & 윈도우 생성
    for r in meta_all.itertuples():
        dom = r.domain; pid = str(r.pid); split = r.split; label = int(r.label)

        # 오디오 LLD
        a_glob = str(audio_dir / f"{dom}__*{pid}*.npy")
        a_files = glob.glob(a_glob)
        if len(a_files)==0: 
            continue
        lld = np.load(a_files[0]) # (T,25)
        # 특권
        priv = None
        if dom in ["edaic","daic"]:
            p_glob = str(priv_dir / f"{dom}__*{pid}*.npy")
            p_files = glob.glob(p_glob)
            if len(p_files)>0:
                priv = np.load(p_files[0])  # (T,Dp)
                # 길이 맞추기
                L = lld.shape[0]
                if priv.shape[0]!=L:
                    priv = resample_to_T(priv, L)
        # 랜드마크/마이크로
        lmk = None; micro = None
        if dom in ["daic","dvlog"]:
            l_glob = str(lmk_dir / f"{dom}__*{pid}*.npy")
            l_files = glob.glob(l_glob)
            if len(l_files)>0:
                lmk = np.load(l_files[0])  # (T,68,2)
                # 평균형상 정렬(선택) — 여기서는 이미 정규화되어 있다고 가정
                L = lld.shape[0]
                if lmk.shape[0]!=L:
                    # 랜드마크 136D로 펼쳐 보간
                    LM = lmk.reshape(lmk.shape[0], -1)
                    LM = resample_to_T(LM, L)
                    lmk = LM.reshape(L,68,2).astype(np.float32)
            if dom=="daic":
                m_glob = str(micro_dir / f"{dom}__*{pid}*.npz")
                m_files = glob.glob(m_glob)
                if len(m_files)>0:
                    z = np.load(m_files[0])
                    micro = np.stack([z["EAR"], z["MAR"], z["head_vel"], z["blink"]], axis=1)  # (T,4)
                    if micro.shape[0]!=lld.shape[0]:
                        micro = resample_to_T(micro, lld.shape[0])

        # 윈도우링
        A = make_windows(lld, win, hop)                 # (Nw, win, 25)
        P = make_windows(priv, win, hop) if priv is not None else None
        if lmk is not None:
            LM = make_windows(lmk.reshape(lmk.shape[0], -1), win, hop)  # (Nw,win,136)
            if micro is not None:
                MI = make_windows(micro, win, hop)                      # (Nw,win,4)
            else:
                MI = None
        else:
            LM = None; MI = None

        # 정규화(zscore) — 글로벌로도 가능하지만 MVP로 윈도우단 간단처리
        if A.shape[0]==0: 
            continue
        # 저장
        for i in range(A.shape[0]):
            pack = {"lld": A[i].astype(np.float32), "label": np.int64(label), "domain": dom, "pid": pid}
            if P is not None:  pack["priv"] = P[i].astype(np.float32)
            if LM is not None: pack["lmk"]  = LM[i].astype(np.float32)
            if MI is not None: pack["micro"]= MI[i].astype(np.float32)
            out_file = out_dir / f"{dom}__{pid}__{i:05d}.npz"
            np.savez_compressed(out_file, **pack)
            index_rows.append({"file": str(out_file), "domain": dom, "pid": pid, "split": split, "label": label})

    idx = pd.DataFrame(index_rows)
    idx.to_csv(Path(env["paths"]["cache_root"]) / env["outputs"]["index_csv"], index=False)
    print("Saved index:", len(idx))
