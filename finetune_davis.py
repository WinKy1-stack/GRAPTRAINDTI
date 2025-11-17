#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune KIBA-trained model on DAVIS dataset
Transfer learning approach for better cross-dataset performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.graphtransdti import GraphTransDTI
from dataloader.davis_loader import DAVISDataset
from dataloader.featurizer import DTIFeaturizer, collate_dti_batch
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index

def convert_kd_to_pkd(kd_values):
    """Convert Kd (nM) to pKd scale"""
    kd_molar = kd_values * 1e-9
    pkd = -np.log10(kd_molar)
    return pkd

def normalize_to_kiba_scale(pkd_values):
    """Normalize pKd to KIBA-like scale (0-15)"""
    min_pkd = 5.0
    max_pkd = 11.0
    normalized = 15 * (pkd_values - min_pkd) / (max_pkd - min_pkd)
    normalized = np.clip(normalized, 0, 15)
    return normalized

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'ci': ci
    }

def normalize_dataset_affinities(dataset):
    """Convert dataset Kd values to normalized scale"""
    original_affinities = []
    for drug_idx, protein_idx, affinity in dataset.pairs:
        original_affinities.append(affinity)
    
    original_affinities = np.array(original_affinities)
    pkd = convert_kd_to_pkd(original_affinities)
    normalized = normalize_to_kiba_scale(pkd)
    
    # Update dataset pairs
    for i in range(len(dataset.pairs)):
        drug_idx, protein_idx, _ = dataset.pairs[i]
        dataset.pairs[i] = (drug_idx, protein_idx, normalized[i])
    
    return dataset, original_affinities, normalized

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        drugs = batch['drug'].to(device)
        proteins = batch['protein'].to(device)
        labels = batch['label'].to(device).squeeze()
        
        optimizer.zero_grad()
        outputs = model(drugs, proteins).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            drugs = batch['drug'].to(device)
            proteins = batch['protein'].to(device)
            labels = batch['label'].to(device).squeeze()
            
            outputs = model(drugs, proteins).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def finetune_on_davis(
    checkpoint_path,
    davis_data_path,
    num_epochs=20,
    batch_size=64,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    early_stopping_patience=5,
    freeze_encoder=False  # Option to freeze drug/protein encoders
):
    """
    Fine-tune KIBA model on DAVIS dataset
    
    Args:
        checkpoint_path: Path to KIBA checkpoint
        davis_data_path: Path to DAVIS data directory
        num_epochs: Number of fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate (typically 10-100x smaller than training)
        early_stopping_patience: Patience for early stopping
        freeze_encoder: If True, only train fusion and predictor layers
    """
    print("="*70)
    print("FINE-TUNING GRAPHTRANSDTI ON DAVIS DATASET")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze encoders: {freeze_encoder}")
    
    # Load datasets
    print("\n[1] Loading DAVIS datasets...")
    featurizer = DTIFeaturizer()
    
    train_dataset = DAVISDataset(
        data_dir=davis_data_path,
        split='train',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        featurizer=featurizer,
        seed=42
    )
    
    val_dataset = DAVISDataset(
        data_dir=davis_data_path,
        split='val',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        featurizer=featurizer,
        seed=42
    )
    
    test_dataset = DAVISDataset(
        data_dir=davis_data_path,
        split='test',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        featurizer=featurizer,
        seed=42
    )
    
    # Normalize affinities
    print("\n[2] Normalizing affinities (Kd → pKd → KIBA scale)...")
    train_dataset, _, _ = normalize_dataset_affinities(train_dataset)
    val_dataset, _, _ = normalize_dataset_affinities(val_dataset)
    test_dataset, test_orig, test_norm = normalize_dataset_affinities(test_dataset)
    
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val:   {len(val_dataset)} pairs")
    print(f"  Test:  {len(test_dataset)} pairs")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_dti_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dti_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dti_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load model from KIBA checkpoint
    print("\n[3] Loading KIBA-trained model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['config']
    model = GraphTransDTI(config=model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  KIBA RMSE: {checkpoint['val_metrics']['rmse']:.4f}")
    
    # Optionally freeze encoder layers
    if freeze_encoder:
        print("\n[4] Freezing drug and protein encoders...")
        for param in model.drug_encoder.parameters():
            param.requires_grad = False
        for param in model.protein_encoder.parameters():
            param.requires_grad = False
        print("  Only training: fusion + predictor layers")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': [],
        'val_pearson': [],
        'val_ci': []
    }
    
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Fine-tuning loop
    print(f"\n[5] Fine-tuning for {num_epochs} epochs...")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['val_ci'].append(val_metrics['ci'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | RMSE: {train_metrics['rmse']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
              f"Pearson: {val_metrics['pearson']:.4f} | CI: {val_metrics['ci']:.4f}")
        
        # Check for improvement
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'config': model_config,
                'training_type': 'finetuned_on_davis'
            }, './checkpoints/GraphTransDTI_DAVIS_finetuned_best.pt')
            
            print(f"✓ Saved best model (RMSE: {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val RMSE: {best_val_rmse:.4f}")
    
    # Load best model for testing
    print("\n[6] Testing on DAVIS test set...")
    checkpoint_best = torch.load('./checkpoints/GraphTransDTI_DAVIS_finetuned_best.pt', 
                                  map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_best['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print("\n" + "="*70)
    print("DAVIS TEST SET RESULTS (FINE-TUNED)")
    print("="*70)
    print(f"MSE:      {test_metrics['mse']:.6f}")
    print(f"RMSE:     {test_metrics['rmse']:.4f}")
    print(f"Pearson:  {test_metrics['pearson']:.4f}")
    print(f"Spearman: {test_metrics['spearman']:.4f}")
    print(f"CI:       {test_metrics['ci']:.4f}")
    print("="*70)
    
    # Save results
    os.makedirs('./results/davis_finetuned', exist_ok=True)
    
    # Save training history
    with open('./results/davis_finetuned/history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Save final results
    results = {
        'fine_tuning': {
            'base_model': 'KIBA checkpoint (epoch 94)',
            'num_epochs': epoch,
            'best_epoch': best_epoch,
            'learning_rate': learning_rate,
            'freeze_encoder': freeze_encoder
        },
        'best_validation': {
            'rmse': float(best_val_rmse),
            'pearson': float(checkpoint_best['val_metrics']['pearson']),
            'ci': float(checkpoint_best['val_metrics']['ci'])
        },
        'test_metrics': {
            'mse': float(test_metrics['mse']),
            'rmse': float(test_metrics['rmse']),
            'pearson': float(test_metrics['pearson']),
            'spearman': float(test_metrics['spearman']),
            'ci': float(test_metrics['ci'])
        },
        'comparison': {
            'kiba_rmse': 0.4615,
            'davis_raw_rmse': 8462.33,
            'davis_normalized_rmse': 10.9078,
            'davis_finetuned_rmse': float(test_metrics['rmse'])
        }
    }
    
    with open('./results/davis_finetuned/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('./results/davis_finetuned/metrics.txt', 'w', encoding='utf-8') as f:
        f.write("DAVIS FINE-TUNED RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write("Fine-tuning Setup:\n")
        f.write(f"  Base model: KIBA checkpoint (epoch 94, RMSE=0.4615)\n")
        f.write(f"  Fine-tuning epochs: {epoch}\n")
        f.write(f"  Best epoch: {best_epoch}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Freeze encoders: {freeze_encoder}\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  MSE: {test_metrics['mse']:.6f}\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  Pearson: {test_metrics['pearson']:.4f}\n")
        f.write(f"  Spearman: {test_metrics['spearman']:.4f}\n")
        f.write(f"  CI: {test_metrics['ci']:.4f}\n\n")
        f.write("Comparison:\n")
        f.write(f"  KIBA (original):       RMSE = 0.4615\n")
        f.write(f"  DAVIS (raw):           RMSE = 8462.33\n")
        f.write(f"  DAVIS (normalized):    RMSE = 10.9078\n")
        f.write(f"  DAVIS (fine-tuned):    RMSE = {test_metrics['rmse']:.4f}\n")
    
    # Visualization
    print("\n[7] Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    ax1 = axes[0, 0]
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Fine-tuning Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE progression
    ax2 = axes[0, 1]
    ax2.plot(epochs_range, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    ax2.plot(epochs_range, history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('RMSE Progression', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pearson & CI progression
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, history['val_pearson'], 'purple', label='Pearson r', linewidth=2)
    ax3.plot(epochs_range, history['val_ci'], 'orange', label='CI', linewidth=2)
    ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Validation Metrics', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Comparison bar chart
    ax4 = axes[1, 1]
    methods = ['KIBA\n(Original)', 'DAVIS\n(Normalized)', 'DAVIS\n(Fine-tuned)']
    rmse_values = [0.4615, 10.9078, test_metrics['rmse']]
    colors = ['green', 'orange', 'blue']
    
    bars = ax4.bar(methods, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('RMSE', fontsize=11)
    ax4.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./results/davis_finetuned/training_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: ./results/davis_finetuned/training_curves.png")
    
    plt.close()
    
    print("\n" + "="*70)
    print("ALL RESULTS SAVED")
    print("="*70)
    print("Files:")
    print("  - checkpoints/GraphTransDTI_DAVIS_finetuned_best.pt")
    print("  - results/davis_finetuned/results.json")
    print("  - results/davis_finetuned/metrics.txt")
    print("  - results/davis_finetuned/history.pkl")
    print("  - results/davis_finetuned/training_curves.png")
    print("="*70)
    
    return test_metrics

if __name__ == '__main__':
    # Configuration
    checkpoint_path = './checkpoints/GraphTransDTI_KIBA_best.pt'
    davis_data_path = './data/davis'
    
    print("\n" + "="*70)
    print("FINE-TUNING OPTIONS")
    print("="*70)
    print("1. Full fine-tuning (all layers trainable)")
    print("2. Partial fine-tuning (freeze encoders, train fusion only)")
    print("="*70)
    
    # Option 1: Full fine-tuning with low learning rate
    print("\n>>> Running FULL fine-tuning...")
    metrics = finetune_on_davis(
        checkpoint_path=checkpoint_path,
        davis_data_path=davis_data_path,
        num_epochs=30,
        batch_size=64,
        learning_rate=1e-5,  # 100x smaller than original training
        early_stopping_patience=7,
        freeze_encoder=False
    )
