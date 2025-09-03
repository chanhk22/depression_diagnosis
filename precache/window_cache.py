# precache/window_cache.py
import os
import glob
import yaml
import numpy as np
import pandas as pd
from features.align import make_target_grid, resample_to_target
from preprocessing.label_mapping import load_labels



class WindowCacheBuilder:
    def __init__(self, config):
        self.config = config
        self.window_len = config['windowing']['length_s']
        self.stride = config['windowing']['stride_s']
        self.base_hz = config['windowing']['base_rate_hz']
        self.min_valid_ratio = config['windowing']['min_valid_ratio']
        self.label_mappings = self._load_all_labels()

    def _load_all_labels(self):
        """Load labels for all datasets"""
        mappings = {}
        
        # DAIC-WOZ labels
        if 'daic_woz' in self.config['labels']:
            labels_dir = os.path.dirname(self.config['labels']['daic_woz']['train_split'])
            mappings['DAIC-WOZ'] = load_labels(labels_dir, dataset_hint='DAIC-WOZ')
            print(f"Loaded {len(mappings['DAIC-WOZ'])} DAIC-WOZ labels")
        
        # E-DAIC labels  
        if 'e_daic' in self.config['labels']:
            labels_dir = os.path.dirname(self.config['labels']['e_daic']['train_split'])
            mappings['E-DAIC'] = load_labels(labels_dir, dataset_hint='E-DAIC')
            print(f"Loaded {len(mappings['E-DAIC'])} E-DAIC labels")
        
        # D-VLOG labels (different structure)
        if 'dvlog' in self.config['labels']:
            dvlog_labels = pd.read_csv(self.config['labels']['dvlog']['labels_csv'])
            mappings['D-VLOG'] = self._process_dvlog_labels(dvlog_labels)
            print(f"Loaded {len(mappings['D-VLOG'])} D-VLOG labels")
        
        return mappings
    

    def _build_dataset_cache(self, dataset_name):
        if dataset_name == "DAIC-WOZ":
            return self._build_daic_cache()
        elif dataset_name == "E-DAIC":
            return self._build_edaic_cache()
        elif dataset_name == "D-VLOG":
            return self._build_dvlog_cache()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # DAIC-WOZ
    def _build_daic_cache(self):
        proc_root = self.config['outputs']['processed_root']
        cache_dir = os.path.join(self.config['outputs']['cache_root'], "DAIC-WOZ")
        os.makedirs(cache_dir, exist_ok=True)

        egemaps_files = glob.glob(f"{proc_root}/DAIC-WOZ/ReEgemaps25LLD/*_egemaps_25lld.csv")
        all_indices = []

        for egemaps_path in egemaps_files:
            session_id = os.path.basename(egemaps_path).split('_')[0]
            modalities = {"audio": egemaps_path}

            clnf_path = f"{proc_root}/DAIC-WOZ/Features/clnf/{session_id}_CLNF_features.txt"
            if os.path.exists(clnf_path):
                modalities["landmarks"] = clnf_path

            covarep_path = f"{proc_root}/DAIC-WOZ/Features/covarep/{session_id}_COVAREP.csv"
            if os.path.exists(covarep_path):
                modalities["covarep"] = covarep_path

            session_index = self._build_session_windows(session_id, modalities, cache_dir, "DAIC-WOZ")
            if session_index is not None:
                all_indices.append(session_index)

        if all_indices:
            combined_df = pd.concat(all_indices, ignore_index=True)
            combined_path = f"{cache_dir}/DAIC-WOZ_all_index.csv"
            combined_df.to_csv(combined_path, index=False)
            return combined_df

        return pd.DataFrame()

    # E-DAIC
    def _build_edaic_cache(self):
        proc_root = self.config['outputs']['processed_root']
        cache_dir = os.path.join(self.config['outputs']['cache_root'], "E-DAIC")       

        os.makedirs(cache_dir, exist_ok=True)

        egemaps_files = glob.glob(f"{proc_root}/E-DAIC/ReEgemaps25LLD/*_egemaps_25lld.csv")
        all_indices = []

        for egemaps_path in egemaps_files:
            session_id = os.path.basename(egemaps_path).split('_')[0]
            modalities = {"audio": egemaps_path}

            for k in ["densenet201", "mfcc", "openface_pose_gaze_au", "vgg16"]:
                path = f"{proc_root}/E-DAIC/Features/{k}/{session_id}_{k}.csv"
                if os.path.exists(path):
                    modalities[k] = path

            session_index = self._build_session_windows_with_privileged(session_id, modalities, cache_dir, "E-DAIC")
            if session_index is not None:
                all_indices.append(session_index)

        if all_indices:
            combined_df = pd.concat(all_indices, ignore_index=True)
            combined_path = f"{cache_dir}/E-DAIC_all_index.csv"
            combined_df.to_csv(combined_path, index=False)
            return combined_df

        return pd.DataFrame()

    # D-VLOG
    def _build_dvlog_cache(self):
        dvlog_root = self.config['paths']['dvlog']['root']
        cache_dir = os.path.join(self.config['outputs']['cache_root'], "D-VLOG")
        os.makedirs(cache_dir, exist_ok=True)

        acoustic_files = glob.glob(f"{dvlog_root}/acoustic/*.npy")
        all_indices = []

        for acoustic_path in acoustic_files:
            session_id = os.path.basename(acoustic_path).split('_')[0]
            modalities = {"audio_npy": acoustic_path}

            visual_path = f"{dvlog_root}/visual/{session_id}.npy"
            if os.path.exists(visual_path):
                modalities["visual_npy"] = visual_path

            session_index = self._build_dvlog_session_windows(session_id, modalities, cache_dir)
            if session_index is not None:
                all_indices.append(session_index)

        if all_indices:
            combined_df = pd.concat(all_indices, ignore_index=True)
            combined_path = f"{cache_dir}/D-VLOG_all_index.csv"
            combined_df.to_csv(combined_path, index=False)
            return combined_df

        return pd.DataFrame()
    
    # Session-level windowing
    def _build_session_windows(self, session_id, modalities, cache_dir, dataset_name):
        timeseries = {}
        for mod_name, mod_path in modalities.items():
            if mod_path.endswith('.csv'):
                timeseries[mod_name] = self._load_csv_timeseries(mod_path)
            elif mod_path.endswith(".txt"):
                timeseries[mod_name] = self._load_csv_timeseries(mod_path, sep=",")
            else:
                continue

        # pick base timeline (prefer audio)
        base_timeline = None
        if "audio" in timeseries and timeseries["audio"][0] is not None:
            base_timeline = timeseries["audio"][0]
        else:
            for _, (t, _, _) in timeseries.items():
                if t is not None:
                    base_timeline = t
                    break

        if base_timeline is None:
            return None

        # window loop
        t_start, t_end = float(base_timeline[0]), float(base_timeline[-1])
        windows, widx, cur = [], 0, t_start

        while cur + self.window_len <= t_end:
            w0, w1 = cur, cur + self.window_len
            target_times = make_target_grid(w0, w1, self.base_hz)

            window_data, valid_audio = {}, 0
            for mod_name, (t, x, cols) in timeseries.items():
                if t is None or x is None:
                    window_data[mod_name] = None
                    continue

                resampled = resample_to_target(t, x, target_times)
                if resampled is not None:
                    window_data[mod_name] = resampled
                    if mod_name == "audio": valid_audio = resampled.shape[0]
                else:
                    window_data[mod_name] = None

            if valid_audio < int(self.min_valid_ratio * self.window_len * self.base_hz):
                cur += self.stride
                continue

            win_path = f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path, **{k: v for k,v in window_data.items() if v is not None})

            windows.append({
                "session": session_id,
                "dataset": dataset_name,
                "w": widx,
                "t0": w0,
                "t1": w1,
                "path": win_path,
                **{f"len_{k}": (v.shape[0] if v is not None else 0) for k,v in window_data.items()}
            })

            widx += 1
            cur += self.stride

        return pd.DataFrame(windows) if windows else None

    def _build_session_windows_with_privileged(self, session_id, modalities, cache_dir, dataset_name):
        audio_t, audio_x, _ = self._load_csv_timeseries(modalities["audio"])
        if audio_t is None: return None

        privileged_data = {}
        for priv in ["densenet201","mfcc","openface_pose_gaze_au","vgg16"]:
            if priv not in modalities: continue
            p = modalities[priv]
            
            if p.endswith('.csv'):
                t,x,_ = self._load_csv_timeseries(p)
                privileged_data[priv] = (t,x)

        t_start, t_end = float(audio_t[0]), float(audio_t[-1])
        windows, widx, cur = [], 0, t_start

        while cur + self.window_len <= t_end:
            w0, w1 = cur, cur + self.window_len
            tgt_times = make_target_grid(w0, w1, self.base_hz)
            audio_res = resample_to_target(audio_t, audio_x, tgt_times)

            if audio_res is None or audio_res.shape[0] < int(self.min_valid_ratio*self.window_len*self.base_hz):
                cur += self.stride; continue

            win_data = {"audio": audio_res}
            for priv,(pt,px) in privileged_data.items():
                if priv in ["vgg16","densenet201"]:
                    win_mean = self._get_window_mean(pt, px, w0, w1)
                    if win_mean is not None: win_data[priv] = win_mean.reshape(1,-1)
                elif priv in ["mfcc","openface_pose_gaze_au"]:
                    res = resample_to_target(pt, px, tgt_times)
                    if res is not None: win_data[priv] = res

            win_path = f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path, **win_data)
            windows.append({
                "session": session_id,
                "dataset": dataset_name,
                "w": widx,
                "t0": w0,"t1":w1,
                "path": win_path,
                **{f"len_{k}": (v.shape[0] if hasattr(v,'shape') else 0) for k,v in win_data.items()}
            })
            widx += 1; cur += self.stride

        return pd.DataFrame(windows) if windows else None

    def _build_dvlog_session_windows(self, session_id, modalities, cache_dir):
        audio = np.load(modalities["audio_npy"])
        visual = None
        if "visual_npy" in modalities:
            visual = np.load(modalities["visual_npy"])
            if visual.ndim==3: visual = visual.reshape(visual.shape[0],-1)

        audio_dur = audio.shape[0]/self.base_hz
        windows,widx,cur = [],0,0.0

        while cur + self.window_len <= audio_dur:
            w0,w1 = cur,cur+self.window_len
            s,e = int(w0*self.base_hz), int(w1*self.base_hz)
            audio_win = audio[s:e]
            if audio_win.shape[0] < int(self.min_valid_ratio*self.window_len*self.base_hz):
                cur+=self.stride; continue

            win_data={"audio":audio_win}
            if visual is not None:
                vis_hz=self.config['preprocessing']['resample']['vis_hz']
                vs,ve=int(w0*vis_hz), int(w1*vis_hz)
                ve=min(ve,visual.shape[0])
                if vs<visual.shape[0]:
                    v_win=visual[vs:ve]
                    if v_win.shape[0]>0:
                        from scipy.interpolate import interp1d
                        st=np.linspace(0,self.window_len,v_win.shape[0])
                        tt=np.linspace(0,self.window_len,audio_win.shape[0])
                        v_interp=np.zeros((audio_win.shape[0],v_win.shape[1]))
                        for i in range(v_win.shape[1]):
                            f=interp1d(st,v_win[:,i],kind='linear',fill_value='extrapolate')
                            v_interp[:,i]=f(tt)
                        win_data["landmarks"]=v_interp.astype(np.float32)

            win_path=f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path,**win_data)
            windows.append({
                "session":session_id,"dataset":"D-VLOG","w":widx,
                "t0":w0,"t1":w1,"path":win_path,
                "len_audio":audio_win.shape[0],
                "len_landmarks":win_data.get("landmarks",np.array([])).shape[0]
            })
            widx+=1; cur+=self.stride

        return pd.DataFrame(windows) if windows else None
    
    # Utils
    # check if has explicit time column
    def _load_csv_timeseries(self, path, assume_hop_s=0.01, **read_kwargs):
        """Load csv file. If no time column, assume fixed hop (e.g., COVAREP).
        - Accepts extra kwargs (e.g., sep=',') and passes them to read_table_smart.
        - Robustly parses time columns that may be numeric, strings, or timedeltas."""
        from preprocessing.utils_io import read_table_smart

        df, _ = read_table_smart(path)
        if df is None or len(df) == 0:
            return None, None, None
        
        for cand in ['frameTime','timestamp',' timestamp', 'time','Time']:
            if cand in df.columns:

                col = df[cand]

                # 1) 숫자/숫자 문자열 우선 시도
                t = None
                try:
                    t_numeric = pd.to_numeric(col, errors='coerce')
                    if t_numeric.notna().any():
                        t = t_numeric.astype(float).values
                except Exception:
                    t = None

                # 2) Timedelta/시간 문자열(예: '0 days 00:00:00') 시도
                if t is None:
                    try:
                        td = pd.to_timedelta(col, errors='coerce')
                        if td.notna().any():
                            t = td.dt.total_seconds().values
                    except Exception:
                        t = None

                # 3) 그래도 실패하면 고정 hop 가정
                if t is None:
                    n_frames = len(df)
                    t = np.arange(n_frames, dtype=float) * assume_hop_s

                x=df.select_dtypes(include=[np.number]).drop(columns=[cand],errors='ignore').values
                cols=df.select_dtypes(include=[np.number]).drop(columns=[cand],errors='ignore').columns.tolist()
                return t,x,cols
            
        #  no explicit time column → assume fixed hop
        n_frames = len(df)
        t = np.arange(n_frames) * assume_hop_s
        x = df.select_dtypes(include=[np.number]).values
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return t, x, cols

    def _get_window_mean(self, t, x, w0, w1):
        if t is None or x is None: return None
        mask=(t>=w0)&(t<w1)
        if not mask.any(): return None
        return np.mean(x[mask],axis=0)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--dataset",choices=['DAIC-WOZ','E-DAIC','D-VLOG','all'],default='all')
    args=parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    builder=WindowCacheBuilder(cfg)

    if args.dataset=='all':
        for d in ['DAIC-WOZ','E-DAIC','D-VLOG']:
            print(f"Building cache for {d}...")
            res=builder._build_dataset_cache(d)
            print(f"✓ {d}: {len(res)} windows")
    else:
        res=builder._build_dataset_cache(args.dataset)
        print(f"✓ {args.dataset}: {len(res)} windows")


if __name__=="__main__":
    main()
