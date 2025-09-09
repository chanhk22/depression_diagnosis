# training/train_teacher_improved.py
import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time

from datasets.window_dataset import WindowDataset
from datasets.base_dataset import BaseDataset
from models.teacher import Teacher
from training.utils_losses import multitask_loss
from training.metrics import classification_metrics, regression_metrics


class TeacherTrainer:
    def __init__(self, config_path, training_config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configs
        with open(config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        with open(training_config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = Teacher(self.model_config["teacher"]).to(self.device)
        self.logger.info(f"Teacher model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config["train"]["lr"],
            weight_decay=self.train_config["train"]["weight_decay"]
        )
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.train_config["train"].get("mixed_precision", True)
        )
        
        # Training state
        self.best_f1 = -1
        self.best_metrics = {}
        
    def setup_logging(self):
        """Setup logging for training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_datasets(self, train_index_path, val_index_path):
        """Load training and validation datasets"""
        try:
            # Use updated WindowDataset that works with .npz cache files
            self.train_dataset = WindowDataset(train_index_path)
            self.val_dataset = WindowDataset(val_index_path)
            
            self.logger.info(f"Loaded training dataset: {len(self.train_dataset)} samples")
            self.logger.info(f"Loaded validation dataset: {len(self.val_dataset)} samples")
            
            # Create data loaders with proper collate function
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config["train"]["batch_size"],
                shuffle=True,
                num_workers=self.train_config["train"]["num_workers"],
                collate_fn=self.collate_fn,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.train_config["train"]["batch_size"],
                shuffle=False,
                num_workers=self.train_config["train"]["num_workers"],
                collate_fn=self.collate_fn,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise
    
    def collate_fn(self, batch):
        """Improved collate function for window data"""
        if not batch:
            return {}
        
        collated = {
            "audio": [],
            "vis": [],
            "priv": {},
            "y_bin": [],
            "y_reg": [],
            "meta": []
        }
        
        # Initialize privileged feature containers
        priv_keys = ["face", "vgg", "densenet", "mfcc", "openface"]
        for key in priv_keys:
            collated["priv"][key] = []
        
        for sample in batch:
            # Audio features (required)
            if sample.get("audio") is not None:
                collated["audio"].append(torch.tensor(sample["audio"], dtype=torch.float32))
            else:
                # Fallback for missing audio
                collated["audio"].append(torch.zeros(400, 25))  # 4s @ 100Hz
            
            # Visual features (optional)
            vis = sample.get("vis")
            if vis is not None:
                if isinstance(vis, np.ndarray):
                    if vis.ndim == 3 and vis.shape[1:] == (68, 2):
                        vis = vis.reshape(vis.shape[0], -1)
                    collated["vis"].append(torch.tensor(vis, dtype=torch.float32))
                else:
                    collated["vis"].append(None)
            else:
                collated["vis"].append(None)
            
            # Privileged features
            priv_data = sample.get("priv", {})
            for key in priv_keys:
                val = priv_data.get(key)
                if val is not None:
                    collated["priv"][key].append(torch.tensor(val, dtype=torch.float32))
                else:
                    collated["priv"][key].append(None)
            
            # Labels
            collated["y_bin"].append(sample.get("y_bin", 0.0))
            collated["y_reg"].append(sample.get("y_reg", 0.0))
            collated["meta"].append(sample.get("meta", {}))
        
        # Pad and stack sequences
        return self.pad_and_stack_batch(collated)
    
    def pad_and_stack_batch(self, collated):
        """Pad sequences to same length and create tensors"""
        batch_size = len(collated["audio"])
        
        # Pad audio sequences
        if collated["audio"]:
            max_audio_len = max(a.shape[0] for a in collated["audio"])
            audio_batch = []
            for audio in collated["audio"]:
                if audio.shape[0] < max_audio_len:
                    pad_len = max_audio_len - audio.shape[0]
                    audio = torch.cat([audio, torch.zeros(pad_len, audio.shape[1])], dim=0)
                audio_batch.append(audio)
            collated["audio"] = torch.stack(audio_batch)
        
        # Handle visual features
        vis_valid = [v for v in collated["vis"] if v is not None]
        if vis_valid:
            max_vis_len = max(v.shape[0] for v in vis_valid)
            vis_batch = []
            for vis in collated["vis"]:
                if vis is None:
                    vis_batch.append(torch.zeros(max_vis_len, 136))
                else:
                    if vis.shape[0] < max_vis_len:
                        pad_len = max_vis_len - vis.shape[0]
                        vis = torch.cat([vis, torch.zeros(pad_len, vis.shape[1])], dim=0)
                    vis_batch.append(vis)
            collated["vis"] = torch.stack(vis_batch)
        else:
            collated["vis"] = None
        
        # Handle privileged features (typically 1D pooled features)
        for key in collated["priv"]:
            valid_features = [f for f in collated["priv"][key] if f is not None]
            if valid_features:
                collated["priv"][key] = torch.stack(valid_features)
            else:
                collated["priv"][key] = None
        
        # Convert labels to tensors
        collated["y_bin"] = torch.tensor(collated["y_bin"], dtype=torch.float32)
        collated["y_reg"] = torch.tensor(collated["y_reg"], dtype=torch.float32)
        
        return collated
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            try:
                # Move data to device
                audio = batch["audio"].to(self.device)
                vis = batch["vis"].to(self.device) if batch["vis"] is not None else None
                priv = {k: (v.to(self.device) if v is not None else None) 
                       for k, v in batch["priv"].items()}
                y_bin = batch["y_bin"].to(self.device)
                y_reg = batch["y_reg"].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.train_config["train"].get("mixed_precision", True)):
                    # Forward pass
                    pred_bin, pred_reg, hidden = self.model(audio, vis, priv)
                    
                    # Compute loss
                    loss, loss_components = multitask_loss(
                        pred_bin, y_bin, pred_reg, y_reg, 
                        self.train_config["loss_weights"]["reg_lambda"]
                    )
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config["train"]["grad_clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                self.logger.error(f"Error in training batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Epoch {epoch} - Average training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        all_preds_bin = []
        all_preds_reg = []
        all_labels_bin = []
        all_labels_reg = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                try:
                    # Move data to device
                    audio = batch["audio"].to(self.device)
                    vis = batch["vis"].to(self.device) if batch["vis"] is not None else None
                    priv = {k: (v.to(self.device) if v is not None else None) 
                           for k, v in batch["priv"].items()}
                    
                    # Forward pass
                    pred_bin, pred_reg, _ = self.model(audio, vis, priv)
                    
                    # Collect predictions and labels
                    all_preds_bin.extend(pred_bin.cpu().numpy())
                    all_preds_reg.extend(pred_reg.cpu().numpy())
                    all_labels_bin.extend(batch["y_bin"].numpy())
                    all_labels_reg.extend(batch["y_reg"].numpy())
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        # Compute metrics
        if all_preds_bin:
            clf_metrics = classification_metrics(all_labels_bin, all_preds_bin)
            reg_metrics = regression_metrics(all_labels_reg, all_preds_reg)
            
            metrics = {**clf_metrics, **reg_metrics}
            return metrics
        else:
            return {}
    
    def save_checkpoint(self, metrics, checkpoint_path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_config": self.model_config,
            "train_config": self.train_config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self, train_index_path, val_index_path, checkpoint_path):
        """Main training loop"""
        self.logger.info("Starting teacher model training...")
        
        # Load datasets
        self.load_datasets(train_index_path, val_index_path)
        
        # Training loop
        for epoch in range(1, self.train_config["train"]["epochs"] + 1):
            start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            metrics = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            # Log results
            if metrics:
                self.logger.info(f"Epoch {epoch} ({epoch_time:.1f}s):")
                self.logger.info(f"  Train Loss: {train_loss:.4f}")
                self.logger.info(f"  Val F1: {metrics.get('f1', 0):.4f}")
                self.logger.info(f"  Val Acc: {metrics.get('acc', 0):.4f}")
                self.logger.info(f"  Val AUC: {metrics.get('auc', 0):.4f}")
                
                # Save best model
                if metrics.get('f1', 0) > self.best_f1:
                    self.best_f1 = metrics['f1']
                    self.best_metrics = metrics.copy()
                    self.save_checkpoint(metrics, checkpoint_path)
                    self.logger.info(f"New best F1: {self.best_f1:.4f}")
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation metrics: {self.best_metrics}")


def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    parser.add_argument('--model_cfg', default='configs/model.yaml', help='Model configuration file')
    parser.add_argument('--train_cfg', default='configs/training.yaml', help='Training configuration file')
    parser.add_argument('--train_index', required=True, help='Training data index CSV')
    parser.add_argument('--val_index', required=True, help='Validation data index CSV')
    parser.add_argument('--ckpt', default='models/checkpoints/teacher_best.pth', help='Checkpoint save path')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = TeacherTrainer(args.model_cfg, args.train_cfg)
    trainer.train(args.train_index, args.val_index, args.ckpt)


if __name__ == "__main__":
    main()