# scripts/check_splits.py
import os
import pandas as pd
import yaml

def check_splits():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cache_root = cfg['outputs']['cache_root']
    
    datasets = ['DAIC-WOZ', 'E-DAIC', 'D-VLOG', 'combined']
    
    print("=== Data Split Summary ===\n")
    
    for dataset in datasets:
        print(f"📊 {dataset}")
        print("-" * 40)
        
        splits_dir = os.path.join(cache_root, dataset, "splits")
        
        if not os.path.exists(splits_dir):
            print(f"  ❌ No splits directory found\n")
            continue
        
        total_windows = 0
        total_sessions = set()
        
        for split_name in ['train', 'dev', 'test', 'valid']:
            split_file = os.path.join(splits_dir, f"{split_name}_index.csv")
            
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                n_windows = len(df)
                n_sessions = df['session'].nunique() if 'session' in df.columns else 0
                split_sessions = set(df['session'].unique()) if 'session' in df.columns else set()
                
                total_windows += n_windows
                total_sessions.update(split_sessions)
                
                print(f"  {split_name:5}: {n_windows:5,} windows, {n_sessions:3} sessions", end="")
                
                if 'y_bin' in df.columns:
                    dep_ratio = df['y_bin'].mean()
                    dep_count = int(df['y_bin'].sum())
                    print(f", dep: {dep_ratio:.3f} ({dep_count}/{n_windows})")
                else:
                    print()
                    
                # Check for session overlap (potential data leakage)
                if split_name != 'train':
                    train_file = os.path.join(splits_dir, "train_index.csv")
                    if os.path.exists(train_file):
                        train_df = pd.read_csv(train_file)
                        if 'session' in train_df.columns and 'session' in df.columns:
                            train_sessions = set(train_df['session'].unique())
                            overlap = split_sessions.intersection(train_sessions)
                            if overlap:
                                print(f"    ⚠️  WARNING: {len(overlap)} sessions overlap with train!")
            else:
                print(f"  {split_name:5}: ❌ Not found")
        
        print(f"  Total: {total_windows:5,} windows, {len(total_sessions):3} unique sessions")
        
        # Dataset distribution for combined
        if dataset == 'combined':
            train_file = os.path.join(splits_dir, "train_index.csv")
            if os.path.exists(train_file):
                df = pd.read_csv(train_file)
                if 'dataset' in df.columns:
                    dataset_dist = df['dataset'].value_counts()
                    print(f"  Datasets: {dict(dataset_dist)}")
        
        # Check for additional files
        other_files = [f for f in os.listdir(splits_dir) 
                      if f.endswith('.csv') and not f.startswith(('train_', 'dev_', 'test_', 'valid_'))]
        if other_files:
            print(f"  Other files: {other_files}")
        
        print()

def check_cache_status():
    print("=== Cache Status ===\n")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cache_root = cfg['outputs']['cache_root']
    
    for dataset in ['DAIC-WOZ', 'E-DAIC', 'D-VLOG']:
        dataset_cache = os.path.join(cache_root, dataset)
        
        if not os.path.exists(dataset_cache):
            print(f"❌ {dataset}: No cache directory")
            continue
            
        # Check index file
        index_file = os.path.join(dataset_cache, f"{dataset}_all_index.csv")
        if os.path.exists(index_file):
            df = pd.read_csv(index_file)
            n_windows = len(df)
            n_sessions = df['session'].nunique() if 'session' in df.columns else 0
            
            # Session range
            if 'session' in df.columns:
                sessions = df['session'].unique()
                session_range = f"{min(sessions)}-{max(sessions)}"
            else:
                session_range = "unknown"
                
            print(f"✅ {dataset}: {n_windows:,} windows from {n_sessions} sessions ({session_range})")
            
            # Check for special columns
            special_cols = []
            if 'official_split' in df.columns:
                special_cols.append("official_split")
            if 'y_bin' in df.columns:
                dep_ratio = df['y_bin'].mean()
                special_cols.append(f"labels (dep: {dep_ratio:.3f})")
            
            if special_cols:
                print(f"  Features: {', '.join(special_cols)}")
        else:
            print(f"⚠️  {dataset}: Cache directory exists but no index file")

def check_data_leakage():
    """Check for potential data leakage between splits"""
    print("=== Data Leakage Check ===\n")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cache_root = cfg['outputs']['cache_root']
    
    datasets_to_check = ['DAIC-WOZ', 'E-DAIC', 'combined']
    
    for dataset in datasets_to_check:
        splits_dir = os.path.join(cache_root, dataset, "splits")
        
        if not os.path.exists(splits_dir):
            continue
            
        print(f"🔍 Checking {dataset} for data leakage...")
        
        # Load all splits
        splits_data = {}
        for split_name in ['train', 'dev', 'test', 'valid']:
            split_file = os.path.join(splits_dir, f"{split_name}_index.csv")
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                if 'session' in df.columns:
                    splits_data[split_name] = set(df['session'].unique())
        
        # Check overlaps
        leakage_found = False
        split_names = list(splits_data.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                split1, split2 = split_names[i], split_names[j]
                overlap = splits_data[split1].intersection(splits_data[split2])
                if overlap:
                    print(f"  ⚠️  LEAKAGE: {len(overlap)} sessions overlap between {split1} and {split2}")
                    if len(overlap) <= 5:
                        print(f"    Sessions: {sorted(list(overlap))}")
                    else:
                        print(f"    Sessions: {sorted(list(overlap))[:5]}...")
                    leakage_found = True
        
        if not leakage_found:
            print(f"  ✅ No data leakage detected")
        
        print()

def print_dataset_overlap():
    """Check for session overlaps between different datasets"""
    print("=== Dataset Overlap Analysis ===\n")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cache_root = cfg['outputs']['cache_root']
    
    dataset_sessions = {}
    
    for dataset in ['DAIC-WOZ', 'E-DAIC', 'D-VLOG']:
        index_file = os.path.join(cache_root, dataset, f"{dataset}_all_index.csv")
        if os.path.exists(index_file):
            df = pd.read_csv(index_file)
            if 'session' in df.columns:
                dataset_sessions[dataset] = set(df['session'].unique())
    
    # Check pairwise overlaps
    datasets = list(dataset_sessions.keys())
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1, ds2 = datasets[i], datasets[j]
            overlap = dataset_sessions[ds1].intersection(dataset_sessions[ds2])
            
            print(f"{ds1} ∩ {ds2}:")
            if overlap:
                print(f"  {len(overlap)} overlapping sessions")
                if len(overlap) <= 10:
                    print(f"  Sessions: {sorted(list(overlap))}")
                else:
                    print(f"  Sample sessions: {sorted(list(overlap))[:10]}...")
            else:
                print(f"  ✅ No overlap (clean separation)")
            print()

if __name__ == "__main__":
    check_cache_status()
    print()
    check_splits()
    check_data_leakage()
    print_dataset_overlap()