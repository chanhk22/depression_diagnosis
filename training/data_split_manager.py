# training/data_split_manager.py
import os
import yaml
import argparse
import pandas as pd

from scripts.resolve_overlap_consistency import main as resolve_overlap_main
from scripts.apply_overlap_resolution import (
    apply_resolution_to_window_cache,
    create_consistent_splits

)
from scripts.create_edaic_splits_clean import create_edaic_splits_from_original


class DataSplitManager:
    def __init__(self, config_path="configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.cache_root = self.config["outputs"]["cache_root"]
        

    '''def _check_existing_splits(self, dataset):
        """Check if splits already exist for dataset"""
        split_dir = os.path.join(self.cache_root, dataset, "splits")
        if not os.path.exists(split_dir):
            return False
        files = [
            os.path.join(split_dir, f"{s}_index.csv") for s in ["train", "dev", "test"]
        ]
        return all(os.path.exists(f) for f in files)'''

    def create_daic_edaic_splits(self, dataset):
        """Run overlap resolution + create splits for DAIC-WOZ / E-DAIC"""
        
        '''if self._check_existing_splits(dataset):
            print(f"ℹ️ {dataset} splits already exist. Skipping.")
            return'''

        print(f"\n=== Resolving {dataset} Overlaps and Creating Splits ===")

        # Step 1: Overlap resolution only once for DAIC-WOZ+E-DAIC
        if dataset == "DAIC-WOZ":
            resolve_overlap_main()
            if apply_resolution_to_window_cache():
                create_consistent_splits()
        elif dataset == "E-DAIC":
            create_edaic_splits_from_original()

    def create_dvlog_splits(self):
        """Create splits for D-VLOG: use fold if available, else random"""
        
        '''if self._check_existing_splits("D-VLOG"):
            print("ℹ️ D-VLOG splits already exist. Skipping.")
            return'''

        index_path = os.path.join(self.cache_root, "D-VLOG", "D-VLOG_all_index.csv")
        if not os.path.exists(index_path):
            print("❌ No D-VLOG index found")
            return

        df = pd.read_csv(index_path)
        sessions = df["session"].unique()
        print(f"Loaded {len(df)} windows from {len(sessions)} sessions in D-VLOG")

        splits_dir = os.path.join(self.cache_root, "D-VLOG", "splits")
        os.makedirs(splits_dir, exist_ok=True)

        if "fold" in df.columns:
            print("ℹ️ Using existing fold assignments in D-VLOG index")
            for split_name in ["train", "valid", "test"]:
                split_df = df[df["fold"].str.lower() == split_name].copy()
                if not split_df.empty:
                    out_path = os.path.join(splits_dir, f"{split_name}_index.csv")
                    split_df.to_csv(out_path, index=False)
                    print(
                        f"  {split_name:5}: {len(split_df):5,} windows, {split_df['session'].nunique()} sessions"
                    )
        else:
            print("ℹ️ No fold column found, using random split instead")

            # Stratified by y_bin if available
            if "y_bin" in df.columns:
                dep_sessions = df[df["y_bin"] == 1]["session"].unique()
                nondep_sessions = df[df["y_bin"] == 0]["session"].unique()
            else:
                dep_sessions, nondep_sessions = sessions, []

            # Simple split ratio 64/16/20
            def split_sessions(sessions, ratio=(0.64, 0.16, 0.20)):
                import numpy as np

                sessions = np.array(sessions)
                np.random.shuffle(sessions)
                n = len(sessions)
                n_train = int(ratio[0] * n)
                n_dev = int(ratio[1] * n)
                return (
                    sessions[:n_train],
                    sessions[n_train : n_train + n_dev],
                    sessions[n_train + n_dev :],
                )

            train_s, dev_s, test_s = split_sessions(sessions)

            split_map = {
                "train": train_s,
                "dev": dev_s,
                "test": test_s,
            }

            for split_name, sess_ids in split_map.items():
                split_df = df[df["session"].isin(sess_ids)].copy()
                out_path = os.path.join(splits_dir, f"{split_name}_index.csv")
                split_df.to_csv(out_path, index=False)
                print(
                    f"  {split_name:5}: {len(split_df):5,} windows, {split_df['session'].nunique()} sessions"
                )

        

    def create_combined_splits(self, datasets=None, output_name="combined"):
        """Combine splits across datasets (default: DAIC-WOZ + E-DAIC)"""
        if datasets is None:
            datasets = ["DAIC-WOZ", "E-DAIC"]

        combined_dir = os.path.join(self.cache_root, output_name, "splits")
        '''if os.path.exists(combined_dir):
            print(f"ℹ️ Combined splits already exist in {combined_dir}. Skipping.")
            return'''

        print(f"\n=== Creating combined splits from {datasets} ===")

        os.makedirs(combined_dir, exist_ok=True)

        for split_name in ["train", "dev", "test"]:
            frames = []
            for dataset in datasets:
                split_path = os.path.join(
                    self.cache_root, dataset, "splits", f"{split_name}_index.csv"
                )
                if os.path.exists(split_path):
                    frames.append(pd.read_csv(split_path))
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined.to_csv(
                    os.path.join(combined_dir, f"{split_name}_index.csv"), index=False
                )
                print(
                    f"  {split_name:5}: {len(combined):5,} windows, {combined['session'].nunique()} sessions"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dataset", choices=["DAIC-WOZ", "E-DAIC", "D-VLOG"], help="Create splits for a specific dataset"
    )
    parser.add_argument("--create-combined", action="store_true")
    args = parser.parse_args()

    manager = DataSplitManager(args.config)

    if args.dataset == "DAIC-WOZ":
        manager.create_daic_edaic_splits("DAIC-WOZ")
    elif args.dataset == "E-DAIC":
        manager.create_daic_edaic_splits("E-DAIC")
    elif args.dataset == "D-VLOG":
        manager.create_dvlog_splits()

    if args.create_combined:
        manager.create_combined_splits()

    print("\n=== Split creation completed ===")


if __name__ == "__main__":
    main()
