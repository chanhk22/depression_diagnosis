# datasets/window_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Dataset for loading pre-cached windows from index CSV files
    Compatible with the new caching system that uses index.csv + .npz files
    """
    
    def __init__(self, index_csv_path, modal_keys=None):
        """
        Args:
            index_csv_path: Path to CSV file with window metadata
            modal_keys: List of modality keys to load (optional)
        """
        self.index_csv_path = index_csv_path
        self.modal_keys = modal_keys or ["audio", "landmarks", "face", "vgg", "densenet", "mfcc", "openface"]
        
        # Load index
        if not os.path.exists(index_csv_path):
            raise FileNotFoundError(f"Index file not found: {index_csv_path}")
            
        self.df = pd.read_csv(index_csv_path)
        print(f"Loaded dataset: {len(self.df)} windows from {self.df['session'].nunique()} sessions")
        
        # Validate required columns
        required_cols = ['path', 'session', 'y_bin', 'y_reg']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in index CSV: {missing_cols}")
        
        # Check file existence for a few samples
        sample_size = min(10, len(self.df))
        missing_files = []
        for _, row in self.df.sample(sample_size).iterrows():
            if not os.path.exists(row['path']):
                missing_files.append(row['path'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} sample files missing (e.g., {missing_files[0]})")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Load and return a single window sample"""
        row = self.df.iloc[idx]
        
        # Load the .npz file
        npz_path = row['path']
        if not os.path.exists(npz_path):
            # Return empty sample if file missing
            return self._create_empty_sample(row)
        
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            sample = self._process_npz_data(npz_data, row)
            return sample
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return self._create_empty_sample(row)
    
    def _process_npz_data(self, npz_data, row):
        """Process loaded npz data into sample format"""
        sample = {
            "meta": {
                "session": row['session'],
                "w": row.get('w', 0),
                "t0": row.get('t0', 0.0),
                "t1": row.get('t1', 4.0),
                "path": row['path']
            }
        }
        
        # Load audio (required)
        audio = npz_data.get('audio')
        if audio is not None:
            sample["audio"] = audio.astype(np.float32)
        else:
            # Fallback: create dummy audio
            sample["audio"] = np.zeros((400, 25), dtype=np.float32)  # 4s @ 100Hz
        
        # Load visual features (landmarks)
        vis = None
        for vis_key in ['landmarks', 'vis']:
            if vis_key in npz_data:
                vis_data = npz_data[vis_key]
                if vis_data.ndim == 3 and vis_data.shape[1:] == (68, 2):
                    # Convert (T, 68, 2) to (T, 136)
                    vis = vis_data.reshape(vis_data.shape[0], -1)
                elif vis_data.ndim == 2 and vis_data.shape[1] == 136:
                    vis = vis_data
                else:
                    print(f"Unexpected visual data shape: {vis_data.shape}")
                break
        
        sample["vis"] = vis.astype(np.float32) if vis is not None else None
        
        # Load privileged features
        priv = {}
        
        # VGG features (pooled to 1 timestep)
        vgg = npz_data.get('vgg')
        if vgg is not None:
            priv["vgg"] = self._pool_to_step1(vgg)
            
        # DenseNet features (pooled to 1 timestep) 
        densenet = npz_data.get('densenet')
        if densenet is not None:
            priv["densenet"] = self._pool_to_step1(densenet)
            
        # Face features (OpenFace pose/gaze/AUs, pooled to 1 timestep)
        for face_key in ['face', 'face_feat', 'openface_pose_gaze_au']:
            if face_key in npz_data:
                face_data = npz_data[face_key]
                priv["face"] = self._pool_to_step1(face_data)
                break
                
        # MFCC features (sequence)
        mfcc = npz_data.get('mfcc')
        if mfcc is not None:
            priv["mfcc"] = mfcc.astype(np.float32)
            
        sample["priv"] = priv
        
        # Labels
        sample["y_bin"] = float(row['y_bin']) if pd.notna(row['y_bin']) else 0.0
        sample["y_reg"] = float(row['y_reg']) if pd.notna(row['y_reg']) else 0.0
        
        return sample
    
    def _pool_to_step1(self, arr):
        """Pool array to single timestep (T,D) -> (1,D)"""
        if arr is None:
            return None
        
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1).astype(np.float32)
        elif arr.ndim == 2:
            return arr.mean(axis=0, keepdims=True).astype(np.float32)
        elif arr.ndim == 3:
            # Flatten spatial dimensions first if needed
            arr_flat = arr.reshape(arr.shape[0], -1)
            return arr_flat.mean(axis=0, keepdims=True).astype(np.float32)
        else:
            return arr.mean(axis=0, keepdims=True).astype(np.float32)
    
    def _create_empty_sample(self, row):
        """Create empty sample when file is missing"""
        return {
            "audio": np.zeros((400, 25), dtype=np.float32),
            "vis": None,
            "priv": {
                "vgg": None,
                "densenet": None, 
                "face": None,
                "mfcc": None
            },
            "y_bin": float(row['y_bin']) if pd.notna(row['y_bin']) else 0.0,
            "y_reg": float(row['y_reg']) if pd.notna(row['y_reg']) else 0.0,
            "meta": {
                "session": row['session'],
                "w": row.get('w', 0),
                "t0": row.get('t0', 0.0), 
                "t1": row.get('t1', 4.0),
                "path": row['path']
            }
        }
    
    def get_dataset_stats(self):
        """Get statistics about the dataset"""
        stats = {
            "total_windows": len(self.df),
            "total_sessions": self.df['session'].nunique(),
            "depression_ratio": self.df['y_bin'].mean() if 'y_bin' in self.df.columns else None,
            "session_range": (self.df['session'].min(), self.df['session'].max()) if 'session' in self.df.columns else None
        }
        
        if 'dataset' in self.df.columns:
            stats["dataset_distribution"] = self.df['dataset'].value_counts().to_dict()
            
        if 'gender' in self.df.columns:
            stats["gender_distribution"] = self.df['gender'].value_counts().to_dict()
            
        return stats