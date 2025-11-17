"""
Test mô hình đã train trên KIBA với DAVIS dataset để đánh giá khả năng generalization
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GraphTransDTI
from dataloader import get_davis_dataloader
from utils.metrics import concordance_index

def evaluate_on_davis(checkpoint_path, device='cuda'):
    """
    Đánh giá model trên DAVIS dataset
    """
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION: KIBA → DAVIS")
    print("="*60)
    
    # Load checkpoint
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = GraphTransDTI(config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    print(f"[INFO] KIBA validation metrics:")
    print(f"  - Loss: {checkpoint['val_loss']:.4f}")
    print(f"  - RMSE: {checkpoint.get('val_rmse', 'N/A')}")
    print(f"  - Pearson: {checkpoint.get('val_pearson', 'N/A')}")
    print(f"  - CI: {checkpoint.get('val_ci', 'N/A')}")
    
    # Load DAVIS dataset
    print(f"\n[INFO] Loading DAVIS dataset...")
    test_loader = get_davis_dataloader(
        batch_size=64,
        split='test',
        num_workers=0
    )
    
    # Count samples
    total_samples = len(test_loader.dataset)
    print(f"[INFO] DAVIS test samples: {total_samples}")
    
    # Get predictions
    print(f"\n[INFO] Running inference on DAVIS...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing on DAVIS"):
            drug_features = batch['drug'].to(device)
            protein_features = batch['protein'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(drug_features, protein_features)
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    pearson, _ = pearsonr(all_labels, all_predictions)
    ci = concordance_index(all_labels, all_predictions)
    
    print("\n" + "="*60)
    print("DAVIS TEST SET RESULTS")
    print("="*60)
    print(f"MSE:     {mse:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"Pearson: {pearson:.4f}")
    print(f"CI:      {ci:.4f}")
    print("="*60)
    
    # Save results
    results = {
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson,
        'ci': ci,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    # Create output directory
    output_dir = './results/davis_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'davis_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CROSS-DATASET EVALUATION: KIBA to DAVIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model trained on: KIBA\n")
        f.write(f"Tested on: DAVIS\n")
        f.write(f"Best epoch: {checkpoint['epoch']}\n\n")
        f.write("DAVIS Test Set Results:\n")
        f.write(f"  MSE:     {mse:.4f}\n")
        f.write(f"  RMSE:    {rmse:.4f}\n")
        f.write(f"  Pearson: {pearson:.4f}\n")
        f.write(f"  CI:      {ci:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"\n[INFO] Metrics saved to: {metrics_file}")
    
    # Plot results
    plot_davis_results(all_predictions, all_labels, rmse, pearson, ci, output_dir)
    
    return results

def plot_davis_results(predictions, labels, rmse, pearson, ci, output_dir):
    """
    Vẽ biểu đồ kết quả test trên DAVIS
    """
    print(f"\n[INFO] Generating DAVIS visualization plots...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter with regression line
    axes[0, 0].scatter(labels, predictions, alpha=0.5, s=20)
    axes[0, 0].plot([labels.min(), labels.max()], [labels.min(), labels.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(labels, predictions, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(labels, p(labels), "b-", alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    axes[0, 0].set_xlabel('True Affinity (DAVIS)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Affinity', fontsize=12)
    axes[0, 0].set_title(f'DAVIS Test Set Predictions\nPearson: {pearson:.4f}, RMSE: {rmse:.4f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = predictions - labels
    axes[0, 1].scatter(labels, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('True Affinity', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (Predicted - True)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 0].axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, 
                       label=f'Mean: {np.mean(residuals):.3f}')
    axes[1, 0].set_xlabel('Prediction Error', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    CROSS-DATASET EVALUATION
    Model: GraphTransDTI
    
    Training Dataset: KIBA
    Test Dataset: DAVIS
    
    Performance Metrics:
    ━━━━━━━━━━━━━━━━━━━━━━
    RMSE:     {rmse:.4f}
    Pearson:  {pearson:.4f}
    CI:       {ci:.4f}
    
    Test Samples: {len(labels):,}
    
    Error Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━
    Mean Error:  {np.mean(residuals):.4f}
    Std Error:   {np.std(residuals):.4f}
    MAE:         {np.mean(np.abs(residuals)):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=14, 
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'davis_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: davis_evaluation.png")
    
    # 2. Detailed comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    from scipy.stats import gaussian_kde
    xy = np.vstack([labels, predictions])
    z = gaussian_kde(xy)(xy)
    
    scatter = axes[0].scatter(labels, predictions, c=z, s=30, alpha=0.6, cmap='viridis')
    axes[0].plot([labels.min(), labels.max()], [labels.min(), labels.max()], 
                 'r--', lw=2, alpha=0.8)
    plt.colorbar(scatter, ax=axes[0], label='Density')
    axes[0].set_xlabel('True Affinity (DAVIS)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Affinity', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Density Scatter Plot\nPearson r = {pearson:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by affinity ranges
    bins = np.percentile(labels, [0, 25, 50, 75, 100])
    bin_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    binned_labels = np.digitize(labels, bins[1:-1])
    
    data_by_bin = [residuals[binned_labels == i] for i in range(len(bin_labels))]
    
    bp = axes[1].boxplot(data_by_bin, labels=bin_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.8)
    axes[1].set_xlabel('Affinity Range', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[1].set_title('Error by Affinity Range', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'davis_detailed_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: davis_detailed_analysis.png")
    print(f"\n[INFO] All DAVIS plots saved to: {output_dir}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    checkpoint_path = './checkpoints/GraphTransDTI_KIBA_best.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[ERROR] Please train the model first!")
        sys.exit(1)
    
    results = evaluate_on_davis(checkpoint_path, device)
