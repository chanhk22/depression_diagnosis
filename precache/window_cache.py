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
            try:
                mappings['DAIC-WOZ'] = load_labels(labels_dir, dataset_hint='DAIC-WOZ')
                print(f"Loaded {len(mappings['DAIC-WOZ'])} DAIC-WOZ labels")
            except Exception as e:
                print(f"Failed to load DAIC-WOZ labels: {e}")
                mappings['DAIC-WOZ'] = {}
        
        # E-DAIC labels  
        if 'e_daic' in self.config['labels']:
            labels_dir = os.path.dirname(self.config['labels']['e_daic']['train_split'])
            try:
                mappings['E-DAIC'] = load_labels(labels_dir, dataset_hint='E-DAIC')
                print(f"Loaded {len(mappings['E-DAIC'])} E-DAIC labels")
            except Exception as e:
                print(f"Failed to load E-DAIC labels: {e}")
                mappings['E-DAIC'] = {}

        # D-VLOG labels (different structure)
        if 'dvlog' in self.config['labels']:
            dvlog_labels_path = self.config['labels']['dvlog']['labels_csv']
            try:
                mappings['D-VLOG'] = load_labels(dvlog_labels_path, dataset_hint='D-VLOG')
                print(f"Loaded {len(mappings['D-VLOG'])} D-VLOG labels")
            except Exception as e:
                print(f"Failed to load D-VLOG labels: {e}")
                mappings['D-VLOG'] = {}
        
        return mappings
    
    def _get_session_labels(self, session_id, dataset_name):
        """Get labels for a specific session"""
        if dataset_name not in self.label_mappings:
            return None, None, None, None
        
        labels = self.label_mappings[dataset_name].get(str(session_id), {})
        return (
            labels.get('PHQ_Score'),
            labels.get('PHQ_Binary'), 
            labels.get('Gender'),
            labels.get('Fold')  # For D-VLOG
        )
    
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

        print(f"Processing {len(egemaps_files)} DAIC-WOZ audio files...")

        for egemaps_path in egemaps_files:
            session_id = os.path.basename(egemaps_path).split('_')[0]

            # Get labels for this session
            phq_score, phq_binary, gender, fold = self._get_session_labels(session_id, "DAIC-WOZ")
            if phq_binary is None:
                print(f"Skipping DAIC session {session_id}: no labels")
                continue
            
            modalities = {"audio": egemaps_path}

            clnf_path = f"{proc_root}/DAIC-WOZ/Features/clnf/{session_id}_CLNF_features.npy"
            if os.path.exists(clnf_path):
                modalities["landmarks"] = clnf_path

            covarep_path = f"{proc_root}/DAIC-WOZ/Features/covarep/{session_id}_COVAREP.csv"
            if os.path.exists(covarep_path):
                modalities["covarep"] = covarep_path

            session_index = self._build_session_windows(
                session_id, modalities, cache_dir, "DAIC-WOZ", 
                phq_score, phq_binary, gender, fold
            )

            if session_index is not None and not session_index.empty:
                all_indices.append(session_index)
                print(f"  Session {session_id}: {len(session_index)} windows")

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

        print(f"Processing {len(egemaps_files)} E-DAIC audio files...")

        for egemaps_path in egemaps_files:
            session_id = os.path.basename(egemaps_path).split('_')[0]

            # Get labels for this session
            phq_score, phq_binary, gender, fold = self._get_session_labels(session_id, "E-DAIC")
            if phq_binary is None:
                print(f"Skipping E-DAIC session {session_id}: no labels")
                continue

            modalities = {"audio": egemaps_path}

            for k in ["densenet201", "mfcc", "openface_pose_gaze_au", "vgg16"]:
                path = f"{proc_root}/E-DAIC/Features/{k}/{session_id}_{k}.csv"
                if os.path.exists(path):
                    modalities[k] = path

            session_index = self._build_session_windows_with_privileged(
                session_id, modalities, cache_dir, "E-DAIC", 
                phq_score, phq_binary, gender, fold
            )
            
            if session_index is not None and not session_index.empty:
                all_indices.append(session_index)
                print(f"  Session {session_id}: {len(session_index)} windows")

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

        print(f"Processing {len(acoustic_files)} D-VLOG acoustic files...")

        for acoustic_path in acoustic_files:
            session_id = os.path.basename(acoustic_path).split('_')[0] if '_' in os.path.basename(acoustic_path) else os.path.splitext(os.path.basename(acoustic_path))[0]

            # Get labels for this session
            phq_score, phq_binary, gender, fold = self._get_session_labels(session_id, "D-VLOG")
            if phq_binary is None:
                print(f"Skipping D-VLOG session {session_id}: no labels")
                continue

            modalities = {"audio_npy": acoustic_path}

            visual_path = f"{dvlog_root}/visual/{session_id}.npy"
            if os.path.exists(visual_path):
                modalities["visual_npy"] = visual_path

            session_index = self._build_dvlog_session_windows(
                session_id, modalities, cache_dir, 
                phq_score, phq_binary, gender, fold
            )

            if session_index is not None and not session_index.empty:
                all_indices.append(session_index)
                print(f"  Session {session_id}: {len(session_index)} windows")

        if all_indices:
            combined_df = pd.concat(all_indices, ignore_index=True)
            combined_path = f"{cache_dir}/D-VLOG_all_index.csv"
            combined_df.to_csv(combined_path, index=False)
            return combined_df

        return pd.DataFrame()
    
    # Session-level windowing (updated with labels)
    def _build_session_windows(self, session_id, modalities, cache_dir, dataset_name, 
                             phq_score, phq_binary, gender, fold):
        """Build windows for DAIC-WOZ (basic features)"""
        timeseries = {}
        for mod_name, mod_path in modalities.items():
            if mod_path.endswith('.csv'):
                timeseries[mod_name] = self._load_csv_timeseries(mod_path)
            elif mod_path.endswith(".txt"):
                timeseries[mod_name] = self._load_csv_timeseries(mod_path)
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
            print(f"  No valid timeline found for session {session_id}")
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
                    continue

                resampled = resample_to_target(t, x, target_times)
                if resampled is not None:
                    window_data[mod_name] = resampled
                    if mod_name == "audio": 
                        valid_audio = resampled.shape[0]

            if valid_audio < int(self.min_valid_ratio * self.window_len * self.base_hz):
                cur += self.stride
                continue

            win_path = f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path, **{k: v for k,v in window_data.items() if v is not None})

            # Create window metadata with labels
            window_meta = {
                "session": session_id,
                "dataset": dataset_name,
                "w": widx,
                "t0": w0,
                "t1": w1,
                "path": win_path,
                "y_reg": phq_score if phq_score is not None else 0.0,
                "y_bin": phq_binary if phq_binary is not None else 0.0,
                "gender": gender if gender is not None else "Unknown",
                "fold": fold if fold is not None else "Unknown"
            }

            # Add modality length info
            for k, v in window_data.items():
                window_meta[f"len_{k}"] = v.shape[0] if v is not None else 0

            windows.append(window_meta)
            widx += 1
            cur += self.stride

        return pd.DataFrame(windows) if windows else None

    def _build_session_windows_with_privileged(self, session_id, modalities, cache_dir, dataset_name,
                                             phq_score, phq_binary, gender, fold):
        """Build windows for E-DAIC (with privileged features)"""
        # Load audio timeline
        audio_t, audio_x, _ = self._load_csv_timeseries(modalities["audio"])
        if audio_t is None: 
            print(f"  No valid audio timeline for session {session_id}")
            return None

        # Load privileged features
        privileged_data = {}
        for priv in ["densenet201","mfcc","openface_pose_gaze_au","vgg16"]:
            if priv not in modalities: 
                continue
            p = modalities[priv]
            
            if p.endswith('.csv'):
                t,x,_ = self._load_csv_timeseries(p)
                if t is not None and x is not None:
                    privileged_data[priv] = (t, x)

        t_start, t_end = float(audio_t[0]), float(audio_t[-1])
        windows, widx, cur = [], 0, t_start

        while cur + self.window_len <= t_end:
            w0, w1 = cur, cur + self.window_len
            tgt_times = make_target_grid(w0, w1, self.base_hz)
            audio_res = resample_to_target(audio_t, audio_x, tgt_times)

            # Check minimum valid ratio
            if audio_res is None or audio_res.shape[0] < int(self.min_valid_ratio*self.window_len*self.base_hz):
                cur += self.stride
                continue

            # Build window data
            win_data = {"audio": audio_res}

            # Add privileged features
            for priv,(pt,px) in privileged_data.items():
                if priv in ["vgg16","densenet201"]:
                    # CNN features: use window mean (1-step features)
                    win_mean = self._get_window_mean(pt, px, w0, w1)
                    if win_mean is not None: 
                        win_data[priv] = win_mean.reshape(1, -1)

                elif priv in ["mfcc","openface_pose_gaze_au"]:
                    # Sequence features: resample to target timeline
                    res = resample_to_target(pt, px, tgt_times)
                    if res is not None: 
                        win_data[priv] = res

            # Save window data
            win_path = f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path, **win_data)

            # create window metadata with labels
            window_meta = {
                "session": session_id,
                "dataset": dataset_name,
                "w": widx,
                "t0": w0,
                "t1": w1,
                "path": win_path,
                "y_reg": phq_score if phq_score is not None else 0.0,
                "y_bin": phq_binary if phq_binary is not None else 0.0,
                "gender": gender if gender is not None else "Unknown",
                "fold": fold if fold is not None else "Unknown"
            }
            # Add modality length info
            for k, v in win_data.items():
                window_meta[f"len_{k}"] = v.shape[0] if hasattr(v, 'shape') else 0

            windows.append(window_meta)
            widx += 1
            cur += self.stride

        return pd.DataFrame(windows) if windows else None

    def _build_dvlog_session_windows(self, session_id, modalities, cache_dir, 
                                   phq_score, phq_binary, gender, fold):
        """Build windows for D-VLOG (pre-extracted numpy arrays)"""
        # Load audio data
        try:
            audio = np.load(modalities["audio_npy"])
        except Exception as e:
            print(f"  Failed to load audio for session {session_id}: {e}")
            return None
        
        # Load visual data if available
        visual = None
        if "visual_npy" in modalities:
            try:
                visual = np.load(modalities["visual_npy"])
                if visual.ndim == 3: 
                    visual = visual.reshape(visual.shape[0], -1)
            except Exception as e:
                print(f"  Failed to load visual for session {session_id}: {e}")

        audio_dur = audio.shape[0]/self.base_hz
        windows,widx,cur = [],0,0.0

        while cur + self.window_len <= audio_dur:
            w0,w1 = cur,cur+self.window_len
            s,e = int(w0*self.base_hz), int(w1*self.base_hz)
            audio_win = audio[s:e]

            # Check minimum valid ratio
            if audio_win.shape[0] < int(self.min_valid_ratio*self.window_len*self.base_hz):
                cur+=self.stride
                continue

            win_data={"audio":audio_win}

            # Add visual data if available
            if visual is not None:
                vis_hz=self.config['preprocessing']['resample']['vis_hz']
                vs,ve=int(w0*vis_hz), int(w1*vis_hz)
                ve=min(ve,visual.shape[0])

                if vs<visual.shape[0]:
                    v_win=visual[vs:ve]
                    if v_win.shape[0]>0:
                        # Interpolate visual to match timeline
                        try:
                            from scipy.interpolate import interp1d
                            st = np.linspace(0, self.window_len, v_win.shape[0])
                            tt = np.linspace(0, self.window_len, audio_win.shape[0])
                            v_interp = np.zeros((audio_win.shape[0], v_win.shape[1]))
                            for i in range(v_win.shape[1]):
                                f = interp1d(st, v_win[:, i], kind='linear', fill_value='extrapolate')
                                v_interp[:, i] = f(tt)
                            win_data["landmarks"] = v_interp.astype(np.float32)
                        except Exception as e:
                            print(f"    Visual interpolation failed for window {widx}: {e}")

            # Save window data
            win_path = f"{cache_dir}/{session_id}_w{widx:05d}.npz"
            np.savez_compressed(win_path, **win_data)


            # Create window metadata with labels
            window_meta = {
                "session": session_id,
                "dataset": "D-VLOG",
                "w": widx,
                "t0": w0,
                "t1": w1,
                "path": win_path,
                "y_reg": phq_score if phq_score is not None else 0.0,
                "y_bin": phq_binary if phq_binary is not None else 0.0,
                "gender": gender if gender is not None else "Unknown",
                "fold": fold if fold is not None else "Unknown",
                "len_audio": audio_win.shape[0],
                "len_landmarks": win_data.get("landmarks", np.array([])).shape[0]
            }
            windows.append(window_meta)
            widx += 1
            cur += self.stride

        return pd.DataFrame(windows) if windows else None
    
    # Utils
    # check if has explicit time column
    def _load_csv_timeseries(self, path, assume_hop_s=0.01, **read_kwargs):
        """Load csv file. If no time column, assume fixed hop (e.g., COVAREP).
        - Accepts extra kwargs (e.g., sep=',') and passes them to read_table_smart.
        - Robustly parses time columns that may be numeric, strings, or timedeltas."""
        from preprocessing.utils_io import read_table_smart

        try:
            df, _ = read_table_smart(path)
            if df is None or len(df) == 0:
                return None, None, None
        except Exception as e:
            print(f"    Failed to read {path}: {e}")
            return None, None, None
        
        # Try to find time column
        time_candidates = ['frameTime', 'timestamp', ' timestamp', 'time', 'Time']
        t = None
        time_col = None
        
        for cand in time_candidates:
            if cand in df.columns:
                time_col = cand
                col = df[cand]

                # Try numeric conversion first
                try:
                    t_numeric = pd.to_numeric(col, errors='coerce')
                    if t_numeric.notna().any():
                        t = t_numeric.astype(float).values
                        break
                except Exception:
                    pass

                # Try timedelta conversion
                try:
                    td = pd.to_timedelta(col, errors='coerce')
                    if td.notna().any():
                        t = td.dt.total_seconds().values
                        break
                except Exception:
                    pass

        # If no time column found, create synthetic time
        if t is None:
            n_frames = len(df)
            t = np.arange(n_frames, dtype=float) * assume_hop_s

        # Extract numeric features (excluding time column)
        if time_col:
            x = df.select_dtypes(include=[np.number]).drop(columns=[time_col], errors='ignore').values
            cols = df.select_dtypes(include=[np.number]).drop(columns=[time_col], errors='ignore').columns.tolist()
        else:
            x = df.select_dtypes(include=[np.number]).values
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return t, x, cols

    def _get_window_mean(self, t, x, w0, w1):
        """Get mean features within window time range"""
        if t is None or x is None: 
            return None
        mask = (t >= w0) & (t < w1)
        if not mask.any(): 
            return None
        return np.mean(x[mask], axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=['DAIC-WOZ','E-DAIC','D-VLOG','all'], default='all')
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    builder = WindowCacheBuilder(cfg)

    if args.dataset == 'all':
        total_windows = 0
        total_sessions = 0
        
        for d in ['DAIC-WOZ', 'E-DAIC', 'D-VLOG']:
            print(f"\n=== Building cache for {d} ===")
            try:
                res = builder._build_dataset_cache(d)
                if not res.empty:
                    n_windows = len(res)
                    n_sessions = res['session'].nunique()
                    total_windows += n_windows
                    total_sessions += n_sessions
                    
                    print(f"✓ {d}: {n_windows} windows from {n_sessions} sessions")
                    
                    if 'y_bin' in res.columns:
                        depression_ratio = res['y_bin'].mean()
                        print(f"  Depression ratio: {depression_ratio:.3f} ({res['y_bin'].sum()}/{len(res)})")
                        
                        if 'gender' in res.columns:
                            gender_dist = res['gender'].value_counts()
                            print(f"  Gender distribution: {gender_dist.to_dict()}")
                else:
                    print(f"✗ {d}: No valid windows generated")
            except Exception as e:
                print(f"✗ {d}: Error occurred - {e}")
        
        print(f"\n=== Summary ===")
        print(f"Total: {total_windows} windows from {total_sessions} sessions across all datasets")
        
    else:
        print(f"\n=== Building cache for {args.dataset} ===")
        try:
            res = builder._build_dataset_cache(args.dataset)
            if not res.empty:
                n_windows = len(res)
                n_sessions = res['session'].nunique()
                print(f"✓ {args.dataset}: {n_windows} windows from {n_sessions} sessions")
                
                if 'y_bin' in res.columns:
                    depression_ratio = res['y_bin'].mean()
                    print(f"  Depression ratio: {depression_ratio:.3f}")
            else:
                print(f"✗ {args.dataset}: No valid windows generated")
        except Exception as e:
            print(f"✗ {args.dataset}: Error occurred - {e}")


if __name__=="__main__":
    main()
