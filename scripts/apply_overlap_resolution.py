# scripts/apply_overlap_resolution.py
import pandas as pd
import yaml
import os

def apply_resolution_to_window_cache():
    """Apply overlap resolution rules to window cache splits"""
    
    # Load resolution rules
    if not os.path.exists("overlap_resolution_rules.csv"):
        print("❌ No resolution rules found. Run resolve_overlap_consistency.py first.")
        return False
    
    resolution_df = pd.read_csv("overlap_resolution_rules.csv")
    print(f"📋 Applying {len(resolution_df)} resolution rules...")
    
    # Load config
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config["outputs"]["cache_root"]
    
    # Apply to each dataset's cached windows
    for dataset in ["DAIC-WOZ", "E-DAIC"]:
        dataset_cache = os.path.join(cache_root, dataset)
        index_file = os.path.join(dataset_cache, f"{dataset}_all_index.csv")
        
        if not os.path.exists(index_file):
            print(f"⚠️  No index file found for {dataset}")
            continue
            
        # Load window index
        windows_df = pd.read_csv(index_file)
        print(f"\n🔄 Processing {dataset}: {len(windows_df)} windows")
        
        # Apply resolution rules
        changes = 0
        for _, rule in resolution_df.iterrows():
            session_id = str(rule["session_id"])
            target_split = rule["resolved_split"]
            
            # Find windows for this session - try multiple ID formats
            session_masks = [
                windows_df["session"].astype(str) == session_id,
                windows_df["session"].astype(str) == str(int(session_id)),
                windows_df["session"] == int(session_id),
                windows_df["session"] == session_id
            ]
            
            session_mask = None
            for mask in session_masks:
                try:
                    if mask.any():
                        session_mask = mask
                        break
                except:
                    continue
            
            if session_mask is not None and session_mask.any():
                # Add resolved split information
                windows_df.loc[session_mask, "official_split"] = target_split
                changes += session_mask.sum()
                print(f"    Session {session_id}: {session_mask.sum()} windows → {target_split}")
        
        if changes > 0:
            # Save updated index
            windows_df.to_csv(index_file, index=False)
            print(f"  ✅ Updated {changes} windows with official split assignments")
        else:
            print(f"  ℹ️  No changes needed for {dataset}")
    
    print(f"\n✅ Resolution rules applied successfully!")
    return True

def create_consistent_splits():
    """Create train/dev/test splits using official split assignments with fallback"""
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config["outputs"]["cache_root"]
    
    for dataset in ["DAIC-WOZ", "E-DAIC"]:
        print(f"\n📊 Creating consistent splits for {dataset}...")
        
        dataset_cache = os.path.join(cache_root, dataset)
        index_file = os.path.join(dataset_cache, f"{dataset}_all_index.csv")
        splits_dir = os.path.join(dataset_cache, "splits")
        
        if not os.path.exists(index_file):
            continue
            
        os.makedirs(splits_dir, exist_ok=True)
        
        # Load windows
        df = pd.read_csv(index_file)
        
        # Check if we have official_split column
        if "official_split" in df.columns:
            print(f"  Using official_split column (covers {df['official_split'].notna().sum()} windows)")
            
            # For sessions with official_split, use that
            # For sessions without official_split, use original dataset splits
            if dataset == "DAIC-WOZ":
                df = assign_original_daic_splits(df, config)
            elif dataset == "E-DAIC":
                df = assign_original_edaic_splits(df, config)
        else:
            print(f"  No official_split column, using original strategy")
            if dataset == "DAIC-WOZ":
                df = assign_original_daic_splits(df, config)
            elif dataset == "E-DAIC":
                df = assign_original_edaic_splits(df, config)
            
        # Create splits using final_split column
        for split_name in ["train", "dev", "test"]:
            split_df = df[df["final_split"] == split_name].copy()
            
            if len(split_df) > 0:
                output_path = os.path.join(splits_dir, f"{split_name}_index.csv")
                split_df.to_csv(output_path, index=False)
                
                n_windows = len(split_df)
                n_sessions = split_df["session"].nunique()
                dep_ratio = split_df["y_bin"].mean() if "y_bin" in split_df.columns else 0
                
                print(f"  {split_name:5}: {n_windows:5,} windows, {n_sessions:3} sessions, dep: {dep_ratio:.3f}")
            else:
                print(f"  {split_name:5}: No data")

def assign_original_daic_splits(df, config):
    """Assign DAIC-WOZ original splits for sessions without official_split"""
    
    # Load original split files
    try:
        train_sessions = set(pd.read_csv(config['labels']['daic_woz']['train_split'])['Participant_ID'].astype(str))
        dev_sessions = set(pd.read_csv(config['labels']['daic_woz']['dev_split'])['Participant_ID'].astype(str))
        test_sessions = set(pd.read_csv(config['labels']['daic_woz']['test_split'])['Participant_ID'].astype(str))
        
        print(f"    Original DAIC splits: train={len(train_sessions)}, dev={len(dev_sessions)}, test={len(test_sessions)}")
    except Exception as e:
        print(f"    Error loading DAIC splits: {e}")
        return df
    
    # Create final_split column
    df['final_split'] = None
    
    for idx, row in df.iterrows():
        session_id = str(row['session'])
        
        # Use official_split if available, otherwise use original
        if pd.notna(row.get('official_split')):
            df.at[idx, 'final_split'] = row['official_split']
        else:
            # Assign based on original splits
            if session_id in train_sessions:
                df.at[idx, 'final_split'] = 'train'
            elif session_id in dev_sessions:
                df.at[idx, 'final_split'] = 'dev'
            elif session_id in test_sessions:
                df.at[idx, 'final_split'] = 'test'
            else:
                print(f"    Warning: Session {session_id} not found in any original split")
    
    return df

def assign_original_edaic_splits(df, config):
    """Assign E-DAIC original splits for sessions without official_split"""
    
    # Load original split files
    try:
        train_sessions = set(pd.read_csv(config['labels']['e_daic']['train_split'])['Participant_ID'].astype(str))
        dev_sessions = set(pd.read_csv(config['labels']['e_daic']['dev_split'])['Participant_ID'].astype(str))
        test_sessions = set(pd.read_csv(config['labels']['e_daic']['test_split'])['Participant_ID'].astype(str))
        
        print(f"    Original E-DAIC splits: train={len(train_sessions)}, dev={len(dev_sessions)}, test={len(test_sessions)}")
    except Exception as e:
        print(f"    Error loading E-DAIC splits: {e}")
        return df
    
    # Create final_split column (E-DAIC shouldn't have official_split, but just in case)
    df['final_split'] = None
    
    for idx, row in df.iterrows():
        session_id = str(row['session'])
        
        # Use official_split if available (shouldn't exist for E-DAIC), otherwise use original
        if pd.notna(row.get('official_split')):
            df.at[idx, 'final_split'] = row['official_split']
        else:
            # Assign based on original splits
            if session_id in train_sessions:
                df.at[idx, 'final_split'] = 'train'
            elif session_id in dev_sessions:
                df.at[idx, 'final_split'] = 'dev'
            elif session_id in test_sessions:
                df.at[idx, 'final_split'] = 'test'
            else:
                print(f"    Warning: Session {session_id} not found in any original E-DAIC split")
    
    return df

def main():
    print("=== Applying Overlap Resolution ===\n")
    
    # Step 1: Apply resolution rules to window cache
    if apply_resolution_to_window_cache():
        # Step 2: Create consistent splits
        create_consistent_splits()
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Verify splits: python scripts/check_splits.py")
        print(f"2. Create combined dataset: python training/data_split_manager.py --create-combined")
        print(f"3. Start training: bash scripts/5_train_models.sh")
    else:
        print(f"\n❌ Failed to apply resolution rules")

if __name__ == "__main__":
    main()