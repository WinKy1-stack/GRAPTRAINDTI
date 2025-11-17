#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GraphTransDTI on DAVIS dataset with normalized affinity values
Converts Kd (nM) to pKd to match KIBA scale
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.graphtransdti import GraphTransDTI
from dataloader.davis_loader import DAVISDataset
from dataloader.featurizer import DTIFeaturizer, collate_dti_batch

def convert_kd_to_pkd(kd_values):
    """
    Convert Kd (nM) to pKd scale
    pKd = -log10(Kd_M) where Kd_M is in Molar
    
    Args:
        kd_values: Kd in nM (nanomolar)
    Returns:
        pKd values (similar scale to KIBA)
    """
    # Convert nM to M (Molar): 1 nM = 1e-9 M
    kd_molar = kd_values * 1e-9
    # Calculate pKd = -log10(Kd)
    pkd = -np.log10(kd_molar)
    return pkd

def normalize_to_kiba_scale(pkd_values):
    """
    Normalize pKd to KIBA-like scale (0-15)
    DAVIS pKd typically ranges from 5-11, we'll map to 0-15
    """
    # Typical DAVIS pKd range
    min_pkd = 5.0  # ~10 μM
    max_pkd = 11.0 # ~0.1 nM
    
    # Normalize to 0-15 (KIBA scale)
    normalized = 15 * (pkd_values - min_pkd) / (max_pkd - min_pkd)
    normalized = np.clip(normalized, 0, 15)  # Ensure within range
    
    return normalized

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson_r,
        'pearson_p': pearson_p,
        'spearman': spearman_r,
        'spearman_p': spearman_p,
        'ci': ci
    }

def test_model_on_davis(checkpoint_path, davis_data_path, batch_size=64):
    """
    Test KIBA-trained model on normalized DAVIS dataset
    """
    print("="*70)
    print("TESTING GRAPHTRANSDTI ON NORMALIZED DAVIS DATASET")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load DAVIS dataset to get test data
    print("\n[1] Loading DAVIS dataset...")
    featurizer = DTIFeaturizer()
    
    # Load full dataset first to get test split
    davis_dataset = DAVISDataset(
        data_dir=davis_data_path,
        split='test',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        featurizer=featurizer,
        seed=42
    )
    
    # Extract original affinities from test set
    print("\n[2] Extracting test affinities...")
    test_affinities_kd = []
    for drug_idx, protein_idx, affinity in davis_dataset.pairs:
        test_affinities_kd.append(affinity)
    
    test_affinities_kd = np.array(test_affinities_kd)
    print(f"  Test samples: {len(test_affinities_kd)}")
    print(f"  Original Kd range: [{np.min(test_affinities_kd):.2f}, {np.max(test_affinities_kd):.2f}] nM")
    
    # Convert Kd to pKd
    print("\n[3] Converting Kd (nM) to pKd...")
    test_affinities_pkd = convert_kd_to_pkd(test_affinities_kd)
    print(f"  pKd range: [{np.min(test_affinities_pkd):.2f}, {np.max(test_affinities_pkd):.2f}]")
    print(f"  pKd mean: {np.mean(test_affinities_pkd):.2f} ± {np.std(test_affinities_pkd):.2f}")
    
    # Normalize to KIBA scale
    print("\n[4] Normalizing to KIBA scale (0-15)...")
    test_affinities_normalized = normalize_to_kiba_scale(test_affinities_pkd)
    print(f"  Normalized range: [{np.min(test_affinities_normalized):.2f}, {np.max(test_affinities_normalized):.2f}]")
    print(f"  Normalized mean: {np.mean(test_affinities_normalized):.2f} ± {np.std(test_affinities_normalized):.2f}")
    
    # Update dataset with normalized affinities
    print("\n[5] Updating dataset with normalized affinities...")
    for i in range(len(davis_dataset.pairs)):
        drug_idx, protein_idx, _ = davis_dataset.pairs[i]
        davis_dataset.pairs[i] = (drug_idx, protein_idx, test_affinities_normalized[i])
    
    test_dataset = davis_dataset
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dti_batch,  # Use custom collate for PyG Data
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    
    # Load model
    print("\n[6] Loading KIBA-trained model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Use config from checkpoint or create default
    if 'config' in checkpoint and checkpoint['config']:
        model_config = checkpoint['config']
    else:
        # Default config matching training setup
        model_config = {
            'drug': {
                'atom_features': 78,
                'edge_features': 12
            },
            'protein': {
                'vocab_size': 26,
                'embedding_dim': 128
            },
            'model': {
                'drug_encoder': {
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'num_heads': 8,
                    'dropout': 0.1,
                    'use_edge_features': True
                },
                'protein_encoder': {
                    'cnn_filters': [128, 128, 128],
                    'kernel_sizes': [3, 5, 7],
                    'lstm_hidden': 128,
                    'lstm_layers': 2,
                    'dropout': 0.1
                },
                'fusion': {
                    'num_heads': 8,
                    'dropout': 0.1
                },
                'predictor': {
                    'hidden_dims': [512, 256],
                    'dropout': 0.1
                }
            }
        }
    
    model = GraphTransDTI(config=model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Original KIBA metrics: RMSE={checkpoint['val_metrics']['rmse']:.4f}")
    
    # Evaluate
    print("\n[7] Running inference on normalized DAVIS...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            drugs = batch['drug'].to(device)
            proteins = batch['protein'].to(device)
            labels = batch['label'].to(device).squeeze()  # Use 'label' key
            
            outputs = model(drugs, proteins).squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n[8] Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("NORMALIZED DAVIS TEST RESULTS")
    print("="*70)
    print(f"MSE:              {metrics['mse']:.6f}")
    print(f"RMSE:             {metrics['rmse']:.4f}")
    print(f"Pearson r:        {metrics['pearson']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"Spearman r:       {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"Concordance Index: {metrics['ci']:.4f}")
    print("="*70)
    
    # Save results
    os.makedirs('./results/davis_normalized', exist_ok=True)
    
    # Save metrics
    with open('./results/davis_normalized/metrics.txt', 'w', encoding='utf-8') as f:
        f.write("DAVIS NORMALIZED TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write("Normalization Method:\n")
        f.write(f"  1. Convert Kd (nM) to pKd: pKd = -log10(Kd * 1e-9)\n")
        f.write(f"  2. Normalize pKd to KIBA scale (0-15)\n\n")
        f.write(f"Data Statistics:\n")
        f.write(f"  Original Kd range: [{np.min(test_affinities_kd):.2f}, {np.max(test_affinities_kd):.2f}] nM\n")
        f.write(f"  pKd range: [{np.min(test_affinities_pkd):.2f}, {np.max(test_affinities_pkd):.2f}]\n")
        f.write(f"  Normalized range: [{np.min(test_affinities_normalized):.2f}, {np.max(test_affinities_normalized):.2f}]\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  MSE: {metrics['mse']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  Pearson r: {metrics['pearson']:.4f} (p={metrics['pearson_p']:.4e})\n")
        f.write(f"  Spearman r: {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})\n")
        f.write(f"  Concordance Index: {metrics['ci']:.4f}\n")
    
    # Save JSON
    results = {
        'normalization': {
            'method': 'Kd (nM) to pKd to KIBA scale (0-15)',
            'original_kd_range': [float(np.min(test_affinities_kd)), float(np.max(test_affinities_kd))],
            'pkd_range': [float(np.min(test_affinities_pkd)), float(np.max(test_affinities_pkd))],
            'normalized_range': [float(np.min(test_affinities_normalized)), float(np.max(test_affinities_normalized))]
        },
        'metrics': {
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'pearson': float(metrics['pearson']),
            'pearson_p': float(metrics['pearson_p']),
            'spearman': float(metrics['spearman']),
            'spearman_p': float(metrics['spearman_p']),
            'ci': float(metrics['ci'])
        },
        'comparison': {
            'kiba_rmse': float(checkpoint['val_metrics']['rmse']),
            'davis_raw_rmse': 8462.33,
            'davis_normalized_rmse': float(metrics['rmse'])
        }
    }
    
    with open('./results/davis_normalized/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Visualization
    print("\n[9] Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot - Predicted vs True
    ax1 = axes[0, 0]
    ax1.scatter(all_labels, all_preds, alpha=0.5, s=10)
    ax1.plot([all_labels.min(), all_labels.max()], 
             [all_labels.min(), all_labels.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True Affinity (Normalized)', fontsize=11)
    ax1.set_ylabel('Predicted Affinity', fontsize=11)
    ax1.set_title(f'DAVIS Normalized Test\nPearson r = {metrics["pearson"]:.4f}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = all_labels - all_preds
    ax2.scatter(all_preds, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Affinity', fontsize=11)
    ax2.set_ylabel('Residuals (True - Predicted)', fontsize=11)
    ax2.set_title(f'Residual Plot\nRMSE = {metrics["rmse"]:.4f}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(all_labels, bins=50, alpha=0.6, label='True', color='blue', edgecolor='black')
    ax3.hist(all_preds, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
    ax3.set_xlabel('Affinity (Normalized Scale)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics comparison
    ax4 = axes[1, 1]
    comparison_data = {
        'KIBA\n(Validation)': [checkpoint['val_metrics']['rmse'], 
                               checkpoint['val_metrics']['pearson'], 
                               checkpoint['val_metrics']['ci']],
        'DAVIS\n(Raw Kd)': [8462.33/1000, -0.3928, 0.3131],  # Scale RMSE for visibility
        'DAVIS\n(Normalized)': [metrics['rmse'], metrics['pearson'], metrics['ci']]
    }
    
    x = np.arange(3)
    width = 0.25
    labels = ['RMSE', 'Pearson r', 'CI']
    
    for i, (dataset, values) in enumerate(comparison_data.items()):
        if i == 1:  # Raw DAVIS - show scaled
            ax4.bar(x + i*width, values, width, label=f'{dataset}\n(RMSE/1000)')
        else:
            ax4.bar(x + i*width, values, width, label=dataset)
    
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Cross-Dataset Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(labels)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('./results/davis_normalized/evaluation.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: ./results/davis_normalized/evaluation.png")
    
    plt.close()
    
    # Create transformation visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original Kd
    axes[0].hist(test_affinities_kd, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Kd (nM)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Original DAVIS Scale\nRange: [{np.min(test_affinities_kd):.0f}, {np.max(test_affinities_kd):.0f}] nM', 
                     fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # pKd
    axes[1].hist(test_affinities_pkd, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('pKd', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Converted to pKd\nRange: [{np.min(test_affinities_pkd):.2f}, {np.max(test_affinities_pkd):.2f}]', 
                     fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Normalized
    axes[2].hist(test_affinities_normalized, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Normalized Affinity', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f'Normalized to KIBA Scale\nRange: [{np.min(test_affinities_normalized):.2f}, {np.max(test_affinities_normalized):.2f}]', 
                     fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('./results/davis_normalized/transformation.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: ./results/davis_normalized/transformation.png")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: ./results/davis_normalized/")
    print(f"  - metrics.txt: Detailed metrics")
    print(f"  - results.json: Machine-readable results")
    print(f"  - evaluation.png: Performance visualization")
    print(f"  - transformation.png: Scale transformation visualization")
    
    return metrics

if __name__ == '__main__':
    checkpoint_path = './checkpoints/GraphTransDTI_KIBA_best.pt'
    davis_data_path = './data/davis'  # Directory path, not pkl file
    
    metrics = test_model_on_davis(
        checkpoint_path=checkpoint_path,
        davis_data_path=davis_data_path,
        batch_size=64
    )
