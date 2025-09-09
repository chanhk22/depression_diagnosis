# scripts/resolve_overlap_consistency.py
import os
import pandas as pd
import glob
import yaml
from collections import defaultdict

def load_all_splits_with_metadata(labels_dir, dataset_name):
    """Load all split files and return unified dataframe with metadata"""
    all_data = []
    
    for split_file in glob.glob(os.path.join(labels_dir, "*.csv")):
        filename = os.path.basename(split_file).lower()
        
        # Determine split type
        if "train" in filename:
            split_type = "train"
        elif "dev" in filename:
            split_type = "dev"  
        elif "test" in filename:
            split_type = "test"
        else:
            continue
            
        df = pd.read_csv(split_file)
        
        # Standardize column names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["participant_id", "id"]:
                col_mapping[col] = "participant_id"
            elif "phq8_binary" in col_lower or "phq_binary" in col_lower:
                col_mapping[col] = "phq_binary"  
            elif "phq8_score" in col_lower or "phq_score" in col_lower:
                col_mapping[col] = "phq_score"
            elif "gender" in col_lower:
                col_mapping[col] = "gender"
                
        df = df.rename(columns=col_mapping)
        
        # Add metadata
        df["split"] = split_type
        df["dataset"] = dataset_name
        df["source_file"] = filename
        
        # Keep only essential columns
        essential_cols = ["participant_id", "phq_binary", "phq_score", "gender", "split", "dataset", "source_file"]
        df = df[[col for col in essential_cols if col in df.columns]]
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def find_overlapping_sessions(daic_df, edaic_df):
    """Find sessions that appear in both datasets"""
    daic_sessions = set(daic_df["participant_id"].dropna())
    edaic_sessions = set(edaic_df["participant_id"].dropna()) 
    
    overlap = daic_sessions.intersection(edaic_sessions)
    
    # Filter to expected range (300-492)
    overlap_filtered = [sid for sid in overlap if str(sid).isdigit() and 300 <= int(sid) <= 492]
    
    return sorted(overlap_filtered)

def analyze_inconsistencies(daic_df, edaic_df, overlap_sessions):
    """Analyze inconsistencies in overlapping sessions"""
    inconsistencies = []
    
    for session_id in overlap_sessions:
        daic_rows = daic_df[daic_df["participant_id"] == session_id]
        edaic_rows = edaic_df[edaic_df["participant_id"] == session_id]
        
        if len(daic_rows) == 0 or len(edaic_rows) == 0:
            continue
            
        # Compare first occurrence (should be only one per dataset)
        daic_row = daic_rows.iloc[0]
        edaic_row = edaic_rows.iloc[0]
        
        issues = []
        
        # Check split consistency
        if daic_row["split"] != edaic_row["split"]:
            issues.append(f"Split mismatch: DAIC={daic_row['split']} vs E-DAIC={edaic_row['split']}")
            
        # Check label consistency  
        if pd.notna(daic_row.get("phq_binary")) and pd.notna(edaic_row.get("phq_binary")):
            if daic_row["phq_binary"] != edaic_row["phq_binary"]:
                issues.append(f"PHQ Binary mismatch: DAIC={daic_row['phq_binary']} vs E-DAIC={edaic_row['phq_binary']}")
                
        if pd.notna(daic_row.get("phq_score")) and pd.notna(edaic_row.get("phq_score")):
            if abs(daic_row["phq_score"] - edaic_row["phq_score"]) > 0.1:
                issues.append(f"PHQ Score mismatch: DAIC={daic_row['phq_score']} vs E-DAIC={edaic_row['phq_score']}")
        
        if issues:
            inconsistencies.append({
                "session_id": session_id,
                "issues": issues,
                "daic_data": daic_row.to_dict(),
                "edaic_data": edaic_row.to_dict()
            })
    
    return inconsistencies

def resolve_strategy_conservative(inconsistencies):
    """
    Conservative strategy: Use most restrictive split assignment
    Priority: test > dev > train (most restrictive first)
    """
    resolution_rules = []
    split_priority = {"test": 3, "dev": 2, "train": 1}
    
    for inc in inconsistencies:
        session_id = inc["session_id"]
        daic_split = inc["daic_data"]["split"]
        edaic_split = inc["edaic_data"]["split"]
        
        # Choose most restrictive split
        if split_priority[daic_split] >= split_priority[edaic_split]:
            final_split = daic_split
            authority = "DAIC-WOZ"
        else:
            final_split = edaic_split  
            authority = "E-DAIC"
            
        resolution_rules.append({
            "session_id": session_id,
            "original_daic_split": daic_split,
            "original_edaic_split": edaic_split, 
            "resolved_split": final_split,
            "authority": authority
        })
    
    return resolution_rules

def main():
    print("=== Analyzing Dataset Overlap Consistency ===\n")
    
    # Load configuration
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    daic_labels_dir = os.path.dirname(config["labels"]["daic_woz"]["train_split"])
    edaic_labels_dir = os.path.dirname(config["labels"]["e_daic"]["train_split"])
    
    # Load data
    print("Loading DAIC-WOZ labels...")
    daic_df = load_all_splits_with_metadata(daic_labels_dir, "DAIC-WOZ")
    print(f"  Found {len(daic_df)} entries from DAIC-WOZ")
    
    print("Loading E-DAIC labels...")  
    edaic_df = load_all_splits_with_metadata(edaic_labels_dir, "E-DAIC")
    print(f"  Found {len(edaic_df)} entries from E-DAIC")
    
    # Find overlaps
    overlap_sessions = find_overlapping_sessions(daic_df, edaic_df)
    print(f"\nFound {len(overlap_sessions)} overlapping sessions (300-492 range)")
    print(f"Overlap sessions: {overlap_sessions[:10]}{'...' if len(overlap_sessions) > 10 else ''}")
    
    # Analyze inconsistencies
    inconsistencies = analyze_inconsistencies(daic_df, edaic_df, overlap_sessions)
    print(f"\nFound {len(inconsistencies)} sessions with inconsistencies")
    
    if inconsistencies:
        print("\n=== Sample Inconsistencies ===")
        for i, inc in enumerate(inconsistencies[:5]):
            print(f"\nSession {inc['session_id']}:")
            for issue in inc['issues']:
                print(f"  - {issue}")
        
        if len(inconsistencies) > 5:
            print(f"\n... and {len(inconsistencies) - 5} more")
        
        # Generate resolution strategy
        print("\n=== Resolution Strategy (Conservative) ===")
        resolution_rules = resolve_strategy_conservative(inconsistencies)
        
        # Summary of resolution
        split_changes = defaultdict(int)
        for rule in resolution_rules:
            if rule["original_daic_split"] != rule["resolved_split"]:
                split_changes[f"DAIC: {rule['original_daic_split']} → {rule['resolved_split']}"] += 1
            if rule["original_edaic_split"] != rule["resolved_split"]:
                split_changes[f"E-DAIC: {rule['original_edaic_split']} → {rule['resolved_split']}"] += 1
        
        print("Split changes needed:")
        for change, count in split_changes.items():
            print(f"  {change}: {count} sessions")
        
        # Save resolution rules
        resolution_df = pd.DataFrame(resolution_rules)
        output_file = "overlap_resolution_rules.csv"
        resolution_df.to_csv(output_file, index=False)
        print(f"\n✅ Resolution rules saved to: {output_file}")
        

        
    else:
        print("✅ No inconsistencies found! Safe to proceed with joint training.")

if __name__ == "__main__":
    main()