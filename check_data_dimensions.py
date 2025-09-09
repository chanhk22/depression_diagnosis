# scripts/check_data_dimensions.py
import pandas as pd
import numpy as np
import yaml

def check_cached_data_dimensions():
    """Check the actual dimensions of cached data"""
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    cache_root = config['outputs']['cache_root']
    
    print("=== Checking Cached Data Dimensions ===\n")
    
    for dataset in ['DAIC-WOZ', 'E-DAIC', 'D-VLOG']:
        splits_dir = f"{cache_root}/{dataset}/splits"
        train_index = f"{splits_dir}/train_index.csv"
        
        if not os.path.exists(train_index):
            print(f"{dataset}: No train index found")
            continue
            
        df = pd.read_csv(train_index)
        print(f"\n📊 {dataset} ({len(df)} windows)")
        print("-" * 30)
        
        # Check a few sample files
        sample_files = df.sample(min(5, len(df)))['path'].tolist()
        
        modality_dims = {}
        
        for npz_path in sample_files:
            if not os.path.exists(npz_path):
                continue
                
            try:
                data = np.load(npz_path, allow_pickle=True)
                
                for key in data.files:
                    arr = data[key]
                    if key not in modality_dims:
                        modality_dims[key] = []
                    modality_dims[key].append(arr.shape)
                    
            except Exception as e:
                print(f"  Error loading {npz_path}: {e}")
        
        # Print dimension summary
        for modality, shapes in modality_dims.items():
            unique_shapes = list(set(shapes))
            if len(unique_shapes) == 1:
                print(f"  {modality:15}: {unique_shapes[0]}")
            else:
                print(f"  {modality:15}: {unique_shapes} (variable)")

if __name__ == "__main__":
    import os
    check_cached_data_dimensions()