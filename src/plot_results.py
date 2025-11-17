"""
Visualization utilities for GraphTransDTI results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history (loss, metrics over epochs)
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 1].plot(history['val_rmse'], color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pearson Correlation
    axes[1, 0].plot(history['val_pearson'], color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Pearson r')
    axes[1, 0].set_title('Validation Pearson Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Concordance Index
    axes[1, 1].plot(history['val_ci'], color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('CI')
    axes[1, 1].set_title('Validation Concordance Index')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved training history plot to {save_path}")
    
    plt.show()


def plot_predictions_vs_true(labels: np.ndarray, 
                             predictions: np.ndarray,
                             title: str = "Predictions vs True Labels",
                             save_path: str = None):
    """
    Scatter plot of predictions vs true labels
    
    Args:
        labels: True labels
        predictions: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(labels, predictions, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Binding Affinity')
    plt.ylabel('Predicted Binding Affinity')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation info
    from scipy import stats
    r, p = stats.pearsonr(labels, predictions)
    plt.text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value = {p:.2e}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved prediction plot to {save_path}")
    
    plt.show()


def plot_residuals(labels: np.ndarray, 
                   predictions: np.ndarray,
                   save_path: str = None):
    """
    Plot residuals (prediction errors)
    
    Args:
        labels: True labels
        predictions: Predicted values
        save_path: Path to save figure
    """
    residuals = predictions - labels
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(predictions, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Binding Affinity')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved residuals plot to {save_path}")
    
    plt.show()


def plot_baseline_comparison(comparison_data: Dict[str, Dict],
                            save_path: str = None):
    """
    Bar plot comparing different models
    
    Args:
        comparison_data: Dict of {model_name: {metric: value}}
        save_path: Path to save figure
    """
    models = list(comparison_data.keys())
    metrics = ['rmse', 'pearson', 'ci']
    metric_names = ['RMSE ↓', 'Pearson r ↑', 'CI ↑']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [comparison_data[model][metric] for model in models]
        
        bars = axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color best model
        if metric == 'rmse':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_color('green')
        
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'Model Comparison - {metric_name}')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved comparison plot to {save_path}")
    
    plt.show()


def create_comparison_table(comparison_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table
    
    Args:
        comparison_data: Dict of {model_name: {metric: value}}
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(comparison_data).T
    df = df[['rmse', 'mse', 'pearson', 'spearman', 'ci']]
    df.columns = ['RMSE', 'MSE', 'Pearson r', 'Spearman ρ', 'CI']
    
    # Format values
    df = df.round(4)
    
    # Highlight best values
    def highlight_best(s):
        if s.name in ['RMSE', 'MSE']:
            is_best = s == s.min()
        else:
            is_best = s == s.max()
        return ['font-weight: bold' if v else '' for v in is_best]
    
    styled = df.style.apply(highlight_best, axis=0)
    
    return df, styled


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting functions...")
    
    # Dummy training history
    history = {
        'train_loss': np.random.rand(50) * 2 + 1,
        'val_loss': np.random.rand(50) * 2 + 1,
        'val_rmse': np.random.rand(50) * 0.5 + 0.5,
        'val_pearson': np.random.rand(50) * 0.2 + 0.7,
        'val_ci': np.random.rand(50) * 0.1 + 0.8
    }
    
    plot_training_history(history)
    
    # Dummy predictions
    labels = np.random.rand(1000) * 10
    predictions = labels + np.random.randn(1000) * 0.5
    
    plot_predictions_vs_true(labels, predictions)
    plot_residuals(labels, predictions)
    
    # Dummy comparison
    comparison = {
        'DeepDTA': {'rmse': 0.45, 'pearson': 0.85, 'ci': 0.88},
        'GraphDTA': {'rmse': 0.42, 'pearson': 0.87, 'ci': 0.89},
        'MolTrans': {'rmse': 0.40, 'pearson': 0.88, 'ci': 0.90},
        'GraphTransDTI': {'rmse': 0.38, 'pearson': 0.90, 'ci': 0.92}
    }
    
    plot_baseline_comparison(comparison)
    
    print("✓ All tests passed!")
