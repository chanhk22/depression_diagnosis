# scripts/debug_privileged_features.py
import os
import glob
import numpy as np
import yaml

def debug_file_existence():
    """Debug if privileged feature files actually exist"""
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    proc_root = config['outputs']['processed_root']
    dvlog_root = config['paths']['dvlog']['root']
    
    print("=== File Existence Debug ===\n")
    
    # Test DAIC-WOZ session 300
    session_id = "300"
    print(f"DAIC-WOZ Session {session_id}:")
    clnf_npy = f"{proc_root}/DAIC-WOZ/Features/clnf/{session_id}_CLNF_features.npy"
    print(f"  CLNF .npy exists: {os.path.exists(clnf_npy)}")
    if os.path.exists(clnf_npy):
        try:
            data = np.load(clnf_npy)
            print(f"    Shape: {data.shape}")
        except Exception as e:
            print(f"    Load error: {e}")
    
    # Test E-DAIC session 600
    session_id = "600"
    print(f"\nE-DAIC Session {session_id}:")
    for feat in ["vgg16", "densenet201", "openface_pose_gaze_au", "mfcc"]:
        path = f"{proc_root}/E-DAIC/Features/{feat}/{session_id}_{feat}.csv"
        exists = os.path.exists(path)
        print(f"  {feat}: {exists}")
        if exists:
            try:
                import pandas as pd
                df = pd.read_csv(path)
                print(f"    Shape: {df.shape}, Columns: {df.columns.tolist()[:3]}...")
            except Exception as e:
                print(f"    Read error: {e}")
    
    # Test D-VLOG
    print(f"\nD-VLOG Files:")
    acoustic_files = glob.glob(f"{dvlog_root}/acoustic/*.npy")[:3]
    for acoustic_path in acoustic_files:
        filename = os.path.basename(acoustic_path)
        if filename.endswith('_acoustic.npy'):
            session_id = filename.replace('_acoustic.npy', '')
        else:
            session_id = os.path.splitext(filename)[0]
        
        visual_path = f"{dvlog_root}/visual/{session_id}_visual.npy"
        print(f"  Session {session_id}:")
        print(f"    Acoustic: {os.path.exists(acoustic_path)}")
        print(f"    Visual: {os.path.exists(visual_path)}")

def debug_cached_windows():
    """Debug what's actually in the cached window files"""
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config['outputs']['cache_root']
    
    print("\n=== Cached Windows Debug ===\n")
    
    for dataset in ["DAIC-WOZ", "E-DAIC", "D-VLOG"]:
        cache_dir = f"{cache_root}/{dataset}"
        if not os.path.exists(cache_dir):
            print(f"{dataset}: No cache directory")
            continue
            
        # Find sample window files
        npz_files = glob.glob(f"{cache_dir}/*.npz")[:3]
        print(f"{dataset}: {len(glob.glob(f'{cache_dir}/*.npz'))} total .npz files")
        
        for npz_file in npz_files:
            filename = os.path.basename(npz_file)
            print(f"  {filename}:")
            
            try:
                data = np.load(npz_file, allow_pickle=True)
                keys = list(data.files)
                print(f"    Keys: {keys}")
                
                for key in keys:
                    arr = data[key]
                    print(f"    {key}: {arr.shape} {arr.dtype}")
                    
            except Exception as e:
                print(f"    Load error: {e}")

def debug_window_cache_execution():
    """Debug the actual execution path of window_cache.py"""
    
    print("\n=== Window Cache Execution Debug ===")
    print("Run this manually in window_cache.py _build_session_windows_with_privileged:")
    print("""
    # Add these debug prints in _build_session_windows_with_privileged:
    
    print(f"DEBUG: modalities keys = {list(modalities.keys())}")
    
    for priv in ["densenet201","mfcc","openface_pose_gaze_au","vgg16"]:
        if priv in modalities:
            p = modalities[priv]
            print(f"DEBUG: Processing {priv} from {p}")
            
            if p.endswith('.csv'):
                t, x, _ = self._load_csv_timeseries_with_cleanup(p, priv)
                if t is not None and x is not None:
                    privileged_data[priv] = (t, x)
                    print(f"DEBUG: Successfully loaded {priv}: {x.shape}")
                else:
                    print(f"DEBUG: Failed to load {priv}")
        else:
            print(f"DEBUG: {priv} not in modalities")
    
    print(f"DEBUG: privileged_data keys = {list(privileged_data.keys())}")
    """)

if __name__ == "__main__":
    debug_file_existence()
    debug_cached_windows()  
    debug_window_cache_execution()