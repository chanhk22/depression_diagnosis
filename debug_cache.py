# scripts/debug_window_cache_issues.py
import os
import glob
import pandas as pd
import numpy as np
import yaml

def debug_daic_features():
    """Debug DAIC-WOZ feature availability"""
    print("=== DAIC-WOZ Feature Debug ===")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    proc_root = config['outputs']['processed_root']
    
    # Check available feature files
    audio_files = glob.glob(f"{proc_root}/DAIC-WOZ/ReEgemaps25LLD/*_egemaps_25lld.csv")
    print(f"Audio files: {len(audio_files)}")
    
    # Check CLNF files
    clnf_txt_files = glob.glob(f"{proc_root}/DAIC-WOZ/Features/clnf/*_CLNF_features.txt")
    clnf_npy_files = glob.glob(f"{proc_root}/DAIC-WOZ/Features/clnf/*_CLNF_features.npy")
    clnf_csv_files = glob.glob(f"{proc_root}/DAIC-WOZ/Features/clnf/*.csv")
    
    print(f"CLNF .txt files: {len(clnf_txt_files)}")
    print(f"CLNF .npy files: {len(clnf_npy_files)}")
    print(f"CLNF .csv files: {len(clnf_csv_files)}")
    
    # Check COVAREP files
    covarep_files = glob.glob(f"{proc_root}/DAIC-WOZ/Features/covarep/*.csv")
    print(f"COVAREP files: {len(covarep_files)}")
    
    # Sample file analysis
    if audio_files:
        sample_audio = audio_files[0]
        session_id = os.path.basename(sample_audio).split('_')[0]
        print(f"\nSample session: {session_id}")
        
        # Check corresponding feature files
        clnf_txt = f"{proc_root}/DAIC-WOZ/Features/clnf/{session_id}_CLNF_features.txt"
        clnf_npy = f"{proc_root}/DAIC-WOZ/Features/clnf/{session_id}_CLNF_features.npy"
        covarep = f"{proc_root}/DAIC-WOZ/Features/covarep/{session_id}_COVAREP.csv"
        
        print(f"  Audio: {os.path.exists(sample_audio)}")
        print(f"  CLNF .txt: {os.path.exists(clnf_txt)}")
        print(f"  CLNF .npy: {os.path.exists(clnf_npy)}")
        print(f"  COVAREP: {os.path.exists(covarep)}")
        
        # Check CLNF content if exists
        if os.path.exists(clnf_txt):
            with open(clnf_txt, 'r') as f:
                header = f.readline().strip()
                print(f"  CLNF header: {header[:100]}...")
                columns = header.split(',')
                print(f"  CLNF columns: {len(columns)} total")
                
                # Check for non-feature columns
                non_feature = [col for col in columns if col.strip().lower() in 
                             ['frame', 'timestamp', ' timestamp', 'confidence', 'success']]
                print(f"  Non-feature columns: {non_feature}")
                print(f"  Feature columns: {len(columns) - len(non_feature)}")

def debug_edaic_features():
    """Debug E-DAIC feature availability"""
    print("\n=== E-DAIC Feature Debug ===")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    proc_root = config['outputs']['processed_root']
    
    # Check available feature files
    audio_files = glob.glob(f"{proc_root}/E-DAIC/ReEgemaps25LLD/*_egemaps_25lld.csv")
    print(f"Audio files: {len(audio_files)}")
    
    # Check privileged feature directories
    for feat_type in ["densenet201", "mfcc", "openface_pose_gaze_au", "vgg16"]:
        feat_dir = f"{proc_root}/E-DAIC/Features/{feat_type}"
        if os.path.exists(feat_dir):
            files = glob.glob(f"{feat_dir}/*.csv")
            print(f"{feat_type} files: {len(files)}")
        else:
            print(f"{feat_type} directory: NOT FOUND")
    
    # Sample file analysis
    if audio_files:
        sample_audio = audio_files[0]
        session_id = os.path.basename(sample_audio).split('_')[0]
        print(f"\nSample session: {session_id}")
        
        for feat_type in ["densenet201", "mfcc", "openface_pose_gaze_au", "vgg16"]:
            feat_file = f"{proc_root}/E-DAIC/Features/{feat_type}/{session_id}_{feat_type}.csv"
            exists = os.path.exists(feat_file)
            print(f"  {feat_type}: {exists}")
            
            if exists and feat_type in ["densenet201", "vgg16", "openface_pose_gaze_au", "mfcc"]:
                # Check CNN feature dimensions
                try:
                    df = pd.read_csv(feat_file)
                    print(f"    Shape: {df.shape}")
                    print(f"    Columns: {df.columns.tolist()[:5]}...")
                except Exception as e:
                    print(f"    Error reading: {e}")

def debug_dvlog_features():
    """Debug D-VLOG feature availability"""
    print("\n=== D-VLOG Feature Debug ===")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    dvlog_root = config['paths']['dvlog']['root']
    
    # Check acoustic files
    acoustic_files = glob.glob(f"{dvlog_root}/acoustic/*.npy")
    print(f"Acoustic files: {len(acoustic_files)}")
    
    # Check visual files
    visual_files = glob.glob(f"{dvlog_root}/visual/*.npy")
    print(f"Visual files: {len(visual_files)}")
    
    # Sample file analysis
    if acoustic_files:
        sample_acoustic = acoustic_files[0]
        session_id = os.path.basename(sample_acoustic).split('.')[0].replace('_acoustic', '')
        print(f"\nSample session: {session_id}")
        
        acoustic_path = f"{dvlog_root}/acoustic/{session_id}.npy"
        visual_path = f"{dvlog_root}/visual/{session_id}.npy"
        
        print(f"  Acoustic: {os.path.exists(acoustic_path)}")
        print(f"  Visual: {os.path.exists(visual_path)}")
        
        if os.path.exists(visual_path):
            try:
                visual_data = np.load(visual_path)
                print(f"  Visual shape: {visual_data.shape}")
            except Exception as e:
                print(f"  Visual load error: {e}")

def check_window_cache_logic():
    """Check window_cache.py logic issues"""
    print("\n=== Window Cache Logic Check ===")
    
    # This would be a manual inspection of the key logic points
    print("Key areas to check in window_cache.py:")
    print("1. DAIC _build_session_windows - landmarks path resolution")
    print("2. E-DAIC _build_session_windows_with_privileged - privileged feature loading")
    print("3. D-VLOG _build_dvlog_session_windows - visual data inclusion")
    print("4. _load_csv_timeseries vs _load_npy_timeseries logic")

if __name__ == "__main__":
    debug_daic_features()
    debug_edaic_features() 
    debug_dvlog_features()
    check_window_cache_logic()