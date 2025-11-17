"""
Training script for GraphTransDTI
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GraphTransDTI, count_parameters
from dataloader import get_kiba_dataloader, get_davis_dataloader
from utils import set_seed, get_device, calculate_metrics, print_metrics
from utils.visualizer import TrainingVisualizer


class Trainer:
    """
    Trainer for GraphTransDTI
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set seed
        set_seed(config['experiment']['seed'])
        
        # Device
        self.device = get_device(prefer_cuda=(config['experiment']['device'] == 'cuda'))
        
        # Model
        self.model = GraphTransDTI(config).to(self.device)
        print(f"\n[INFO] Model parameters: {count_parameters(self.model):,}")
        
        # Loss function
        if config['loss']['type'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss']['type'] == 'huber':
            self.criterion = nn.HuberLoss()
        elif config['loss']['type'] == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        
        # Optimizer
        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        
        # Scheduler
        if config['training']['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs']
            )
        
        # Data loaders
        dataset_name = config['data']['dataset']
        if dataset_name == 'kiba':
            self.train_loader = get_kiba_dataloader(
                data_dir=os.path.join(config['data']['data_dir'], 'kiba'),
                split='train',
                batch_size=config['training']['batch_size'],
                num_workers=config['data']['num_workers'],
                shuffle=True,
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                test_ratio=config['data']['test_ratio'],
                seed=config['experiment']['seed']
            )
            
            self.val_loader = get_kiba_dataloader(
                data_dir=os.path.join(config['data']['data_dir'], 'kiba'),
                split='val',
                batch_size=config['training']['batch_size'],
                num_workers=config['data']['num_workers'],
                shuffle=False,
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                test_ratio=config['data']['test_ratio'],
                seed=config['experiment']['seed']
            )
        elif dataset_name == 'davis':
            self.train_loader = get_davis_dataloader(
                data_dir=os.path.join(config['data']['data_dir'], 'davis'),
                split='train',
                batch_size=config['training']['batch_size'],
                num_workers=config['data']['num_workers'],
                shuffle=True,
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                test_ratio=config['data']['test_ratio'],
                seed=config['experiment']['seed']
            )
            
            self.val_loader = get_davis_dataloader(
                data_dir=os.path.join(config['data']['data_dir'], 'davis'),
                split='val',
                batch_size=config['training']['batch_size'],
                num_workers=config['data']['num_workers'],
                shuffle=False,
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                test_ratio=config['data']['test_ratio'],
                seed=config['experiment']['seed']
            )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_pearson': [],
            'val_ci': []
        }
        
        # Checkpoint directory
        os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer()
        self.save_plots_every_epoch = config['training'].get('save_plots_every_epoch', False)
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if batch is None:
                continue
            
            # Move to device
            drug_batch = batch['drug'].to(self.device)
            protein_seq = batch['protein'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            predictions = self.model(drug_batch, protein_seq)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if batch is None:
                    continue
                
                # Move to device
                drug_batch = batch['drug'].to(self.device)
                protein_seq = batch['protein'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = self.model(drug_batch, protein_seq)
                
                # Compute loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Store for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        metrics = calculate_metrics(all_labels, all_predictions)
        
        return avg_loss, metrics, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"{'='*60}")      
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_metrics, val_predictions, val_labels = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            print_metrics(val_metrics, prefix="Validation")
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_rmse'].append(val_metrics['rmse'])
            self.training_history['val_pearson'].append(val_metrics['pearson'])
            self.training_history['val_ci'].append(val_metrics['ci'])
            
            # Auto-generate plots if enabled
            if self.save_plots_every_epoch:
                self.visualizer.update_plots(
                    self.training_history, 
                    epoch,
                    val_predictions,
                    val_labels
                )
            self.training_history['val_rmse'].append(val_metrics['rmse'])
            self.training_history['val_pearson'].append(val_metrics['pearson'])
            self.training_history['val_ci'].append(val_metrics['ci'])
            
            # Scheduler step
            if self.config['training']['scheduler'] == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(
                    self.config['experiment']['checkpoint_dir'],
                    f"{self.config['experiment']['name']}_best.pt"
                )
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, checkpoint_path)
                
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
                print(f"[INFO] Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
                break
        
        print("\n" + "=" * 60)
        print("Training Completed")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        return self.training_history


def main():
    """Main training function"""
    # Load config
    config_path = "./config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 60)
    print("GraphTransDTI Training")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Dataset: {config['data']['dataset'].upper()}")
    print(f"Device: {config['experiment']['device']}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train
    history = trainer.train()
    
    # Save training history
    history_path = os.path.join(
        config['experiment']['checkpoint_dir'],
        f"{config['experiment']['name']}_history.pkl"
    )
    import pickle
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n[INFO] Training history saved to {history_path}")


if __name__ == "__main__":
    main()
