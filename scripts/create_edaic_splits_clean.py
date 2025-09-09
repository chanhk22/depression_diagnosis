# scripts/create_edaic_splits_clean.py
import pandas as pd
import yaml
import os

def create_edaic_splits_from_original():
    """E-DAIC uses original official splits since no overlap with DAIC-WOZ"""
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config["outputs"]["cache_root"]
    
    # Load E-DAIC window index
    edaic_index = os.path.join(cache_root, "E-DAIC", "E-DAIC_all_index.csv")
    if not os.path.exists(edaic_index):
        print("E-DAIC index file not found")
        return
    
    windows_df = pd.read_csv(edaic_index)
    print(f"Processing E-DAIC: {len(windows_df)} windows")
    
    # Load original E-DAIC splits
    try:
        train_sessions = set(pd.read_csv(config['labels']['e_daic']['train_split'])['Participant_ID'].astype(str))
        dev_sessions = set(pd.read_csv(config['labels']['e_daic']['dev_split'])['Participant_ID'].astype(str))
        test_sessions = set(pd.read_csv(config['labels']['e_daic']['test_split'])['Participant_ID'].astype(str))
        
        print(f"Original splits - Train: {len(train_sessions)}, Dev: {len(dev_sessions)}, Test: {len(test_sessions)}")
        
    except Exception as e:
        print(f"Error loading original splits: {e}")
        return
    
    # Create splits directory
    splits_dir = os.path.join(cache_root, "E-DAIC", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Assign windows to splits based on session membership
    for split_name, session_set in [("train", train_sessions), ("dev", dev_sessions), ("test", test_sessions)]:
        split_windows = []
        
        for _, row in windows_df.iterrows():
            session_id = str(row['session'])
            if session_id in session_set:
                split_windows.append(row)
        
        if split_windows:
            split_df = pd.DataFrame(split_windows)
            output_path = os.path.join(splits_dir, f"{split_name}_index.csv")
            split_df.to_csv(output_path, index=False)
            
            n_windows = len(split_df)
            n_sessions = split_df['session'].nunique()
            dep_ratio = split_df['y_bin'].mean() if 'y_bin' in split_df.columns else 0
            
            print(f"  {split_name:5}: {n_windows:5,} windows, {n_sessions:3} sessions, dep: {dep_ratio:.3f}")
        else:
            print(f"  {split_name:5}: No windows found")

if __name__ == "__main__":
    create_edaic_splits_from_original()