# scripts/debug_session_loss.py
import pandas as pd
import yaml
import os

def debug_session_loss():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config['outputs']['cache_root']
    
    print("=== Debugging Session Loss ===\n")
    
    # Check individual dataset caches
    for dataset in ['DAIC-WOZ', 'E-DAIC']:
        print(f"📊 {dataset} Cache Analysis")
        print("-" * 40)
        
        index_path = os.path.join(cache_root, dataset, f"{dataset}_all_index.csv")
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            total_sessions = df['session'].nunique()
            total_windows = len(df)
            
            print(f"Cache: {total_windows} windows from {total_sessions} sessions")
            
            # Check session range
            sessions = sorted(df['session'].unique())
            print(f"Session range: {min(sessions)} - {max(sessions)}")
            
            # Check for labels
            if 'y_bin' in df.columns:
                labeled_sessions = df[df['y_bin'].notna()]['session'].nunique()
                labeled_windows = df[df['y_bin'].notna()].shape[0]
                print(f"With labels: {labeled_windows} windows from {labeled_sessions} sessions")
                
                if labeled_sessions < total_sessions:
                    unlabeled = set(df['session'].unique()) - set(df[df['y_bin'].notna()]['session'].unique())
                    print(f"⚠️  {len(unlabeled)} sessions without labels: {sorted(list(unlabeled))[:10]}...")
            
            print()
        else:
            print(f"❌ No cache found")
            print()
    
    # Check individual splits
    print("=== Individual Dataset Splits ===\n")
    
    for dataset in ['DAIC-WOZ', 'E-DAIC']:
        print(f"📊 {dataset} Splits")
        print("-" * 30)
        
        splits_dir = os.path.join(cache_root, dataset, "splits")
        if not os.path.exists(splits_dir):
            print(f"❌ No splits directory")
            continue
        
        total_split_sessions = set()
        for split_name in ['train', 'dev', 'test']:
            split_path = os.path.join(splits_dir, f"{split_name}_index.csv")
            if os.path.exists(split_path):
                df = pd.read_csv(split_path)
                sessions = set(df['session'].unique())
                total_split_sessions.update(sessions)
                
                print(f"{split_name:5}: {len(df):6,} windows, {len(sessions):3} sessions")
            else:
                print(f"{split_name:5}: ❌ Not found")
        
        print(f"Total: {len(total_split_sessions)} unique sessions across all splits")
        print()
    
    # Check combined splits details
    print("=== Combined Splits Analysis ===\n")
    
    combined_dir = os.path.join(cache_root, "combined", "splits")
    if os.path.exists(combined_dir):
        for split_name in ['train', 'dev', 'test']:
            split_path = os.path.join(combined_dir, f"{split_name}_index.csv")
            if os.path.exists(split_path):
                df = pd.read_csv(split_path)
                print(f"{split_name.upper()} Split Analysis:")
                print(f"  Total: {len(df)} windows from {df['session'].nunique()} sessions")
                
                if 'dataset' in df.columns:
                    dataset_dist = df['dataset'].value_counts()
                    print(f"  Dataset distribution: {dict(dataset_dist)}")
                    
                    # Session distribution by dataset
                    for ds in dataset_dist.index:
                        ds_sessions = df[df['dataset'] == ds]['session'].nunique()
                        print(f"    {ds}: {ds_sessions} sessions")
                
                # Sample sessions
                sample_sessions = sorted(df['session'].unique())[:10]
                print(f"  Sample sessions: {sample_sessions}")
                print()
    
    # Check for potential filtering issues
    print("=== Potential Issues Check ===\n")
    
    # Check if y_bin filtering is too aggressive
    for dataset in ['DAIC-WOZ', 'E-DAIC']:
        index_path = os.path.join(cache_root, dataset, f"{dataset}_all_index.csv")
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            
            print(f"{dataset} Label Analysis:")
            if 'y_bin' in df.columns:
                print(f"  y_bin column exists: {df['y_bin'].notna().sum()}/{len(df)} non-null")
                print(f"  y_bin values: {df['y_bin'].value_counts().to_dict()}")
                
                # Check for NaN labels by session
                session_labels = df.groupby('session')['y_bin'].first()
                null_sessions = session_labels[session_labels.isna()].index.tolist()
                if null_sessions:
                    print(f"  Sessions with NaN labels: {len(null_sessions)}")
                    print(f"    Examples: {null_sessions[:5]}")
            else:
                print(f"  ❌ No y_bin column found")
            print()

def check_label_consistency():
    """Check if labels are being filtered out somewhere"""
    print("=== Label Consistency Check ===\n")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config['outputs']['cache_root']
    
    for dataset in ['DAIC-WOZ', 'E-DAIC']:
        print(f"Checking {dataset} label pipeline...")
        
        # Check cache
        index_path = os.path.join(cache_root, dataset, f"{dataset}_all_index.csv")
        if os.path.exists(index_path):
            cache_df = pd.read_csv(index_path)
            cache_sessions_with_labels = cache_df[cache_df['y_bin'].notna()]['session'].nunique()
            
            print(f"  Cache: {cache_sessions_with_labels} sessions with labels")
            
            # Check individual splits
            splits_dir = os.path.join(cache_root, dataset, "splits")
            if os.path.exists(splits_dir):
                split_sessions = set()
                for split_name in ['train', 'dev', 'test']:
                    split_path = os.path.join(splits_dir, f"{split_name}_index.csv")
                    if os.path.exists(split_path):
                        split_df = pd.read_csv(split_path)
                        split_sessions.update(split_df['session'].unique())
                
                print(f"  Splits: {len(split_sessions)} total sessions")
                
                if cache_sessions_with_labels != len(split_sessions):
                    print(f"  ⚠️  MISMATCH: Cache has {cache_sessions_with_labels} labeled sessions, splits have {len(split_sessions)}")
            
            print()

if __name__ == "__main__":
    debug_session_loss()
    check_label_consistency()