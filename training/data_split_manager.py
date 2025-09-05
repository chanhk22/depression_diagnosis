# training/data_split_manager.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocessing.label_mapping import load_labels

class DataSplitManager:
    def __init__(self, config):
        self.config = config
        self.cache_root = config['outputs']['cache_root']
        
    def create_unified_splits(self):
        """Create unified train/val/test splits across DAIC-WOZ and E-DAIC"""
        print(self.config['labels']['daic_woz'])
        # 1. Load all cached indices
        daic_idx = self._load_dataset_index("DAIC-WOZ")
        edaic_idx = self._load_dataset_index("E-DAIC") 
        dvlog_idx = self._load_dataset_index("D-VLOG")
        
        # 2. Load labels for DAIC + E-DAIC
        daic_labels = load_labels(self.config['labels']['daic_woz'])
        edaic_labels = load_labels(self.config['labels']['e_daic'])
        
        # 3. Merge indices with labels
        daic_merged = self._merge_with_labels(daic_idx, daic_labels, "DAIC-WOZ")
        edaic_merged = self._merge_with_labels(edaic_idx, edaic_labels, "E-DAIC")
        
        # 4. Combine DAIC + E-DAIC for supervised learning
        supervised_df = pd.concat([daic_merged, edaic_merged], ignore_index=True)
        
        # 5. Stratified split by depression + gender
        train_df, test_df = self._stratified_split(
            supervised_df, test_size=0.2, random_state=42
        )
        train_df, val_df = self._stratified_split(
            train_df, test_size=0.2, random_state=42  # 0.64/0.16/0.2 split
        )
        
        # 6. Add D-VLOG as unlabeled domain adaptation data
        dvlog_merged = self._prepare_dvlog(dvlog_idx)
        
        # 7. Save splits
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'dvlog_unlabeled': dvlog_merged
        }
        
        for split_name, df in splits.items():
            out_path = f"{self.cache_root}/unified_{split_name}_index.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {split_name}: {len(df)} samples -> {out_path}")
        
        return splits
    
    def _load_dataset_index(self, dataset_name):
        """Load concatenated index for a dataset"""
        idx_path = f"{self.cache_root}/{dataset_name}/{dataset_name}_all_index.csv"
        if os.path.exists(idx_path):
            return pd.read_csv(idx_path)
        else:
            # Fallback: concat individual session indices
            session_indices = []
            for f in os.listdir(f"{self.cache_root}/{dataset_name}"):
                if f.endswith("_index.csv") and not f.endswith("_all_index.csv"):
                    df = pd.read_csv(f"{self.cache_root}/{dataset_name}/{f}")
                    df['dataset'] = dataset_name
                    session_indices.append(df)
            return pd.concat(session_indices, ignore_index=True) if session_indices else pd.DataFrame()
    
    def _merge_with_labels(self, index_df, label_dict, dataset_name):
        """Merge window index with labels"""
        if index_df.empty:
            return index_df
        
        # Add labels to each window
        index_df['dataset'] = dataset_name
        index_df['y_bin'] = None
        index_df['y_reg'] = None
        index_df['gender'] = None
        
        for idx, row in index_df.iterrows():
            session_id = str(row['session'])
            if session_id in label_dict:
                labels = label_dict[session_id]
                index_df.at[idx, 'y_bin'] = labels.get('PHQ_Binary')
                index_df.at[idx, 'y_reg'] = labels.get('PHQ_Score')
                index_df.at[idx, 'gender'] = labels.get('Gender')
        
        # Filter out samples without labels
        return index_df.dropna(subset=['y_bin']).reset_index(drop=True)
    
    def _prepare_dvlog(self, dvlog_idx):
        """Prepare D-VLOG as unlabeled data for domain adaptation"""
        if dvlog_idx.empty:
            return dvlog_idx
        
        dvlog_labels = pd.read_csv(self.config['labels']['dvlog']['labels_csv'])
        merged = dvlog_idx.merge(dvlog_labels, left_on='session', right_on='index', how='inner')
        
        # Binary conversion for D-VLOG
        merged['y_bin'] = (merged['label'] == 'depression').astype(int)
        merged['y_reg'] = 0.0  # No PHQ scores in D-VLOG
        merged['dataset'] = 'D-VLOG'
        
        return merged[['session', 'w', 't0', 't1', 'path', 'y_bin', 'y_reg', 'dataset']]
    
    def _stratified_split(self, df, test_size=0.2, random_state=42):
        """Stratified split by depression + gender"""
        # Create stratification key
        df['strat_key'] = df['y_bin'].astype(str) + '_' + df['gender'].fillna('Unknown')
        
        # Session-level split (not window-level to avoid data leakage)
        sessions = df.groupby('session').first().reset_index()
        
        train_sessions, test_sessions = train_test_split(
            sessions['session'].values,
            test_size=test_size,
            random_state=random_state,
            stratify=sessions['strat_key'].values
        )
        
        train_df = df[df['session'].isin(train_sessions)].copy()
        test_df = df[df['session'].isin(test_sessions)].copy()
        
        return train_df.drop('strat_key', axis=1), test_df.drop('strat_key', axis=1)

# Usage script
def main():
    import yaml
    with open("configs/default.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 경로 변환: 환경 변수 치환
    def expand_paths(config):
        for section in config:
            if isinstance(config[section], dict):
                for key, value in config[section].items():
                    if isinstance(value, str) and "${" in value:
                        config[section][key] = os.path.expandvars(value)
                    elif isinstance(value, dict):
                        expand_paths(config[section])  # 재귀적으로 처리
        return config

# 경로 치환
    config = expand_paths(config)
    
    manager = DataSplitManager(config)
    splits = manager.create_unified_splits()
    
    print("Data split summary:")
    for name, df in splits.items():
        if not df.empty:
            print(f"{name}: {len(df)} windows, {df['session'].nunique()} sessions")
            if 'y_bin' in df.columns:
                print(f"  Depression ratio: {df['y_bin'].mean():.3f}")

if __name__ == "__main__":
    main()