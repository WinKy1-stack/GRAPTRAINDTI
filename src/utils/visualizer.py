"""
Auto-visualization callback for training
Tự động tạo biểu đồ sau mỗi epoch
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """Tự động tạo biểu đồ trong quá trình training"""
    
    def __init__(self, output_dir='./results/training_progress'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'epoch_plots'), exist_ok=True)
        
    def update_plots(self, history, epoch, val_predictions=None, val_labels=None):
        """
        Cập nhật tất cả biểu đồ sau mỗi epoch
        
        Args:
            history: Training history dict
            epoch: Current epoch number
            val_predictions: Validation predictions (optional)
            val_labels: Validation labels (optional)
        """
        print(f"\n[Visualizer] Updating plots for epoch {epoch}...")
        
        # 1. Live training curves
        self._plot_live_training_curves(history, epoch)
        
        # 2. Prediction scatter (nếu có predictions)
        if val_predictions is not None and val_labels is not None:
            self._plot_prediction_scatter(val_predictions, val_labels, epoch)
            self._plot_correlation_matrix(val_predictions, val_labels, epoch)
            self._plot_error_heatmap(val_predictions, val_labels, epoch)
        
        # 3. Learning rate schedule
        self._plot_learning_rate(history, epoch)
        
        print(f"[Visualizer] ✓ Plots saved to {self.output_dir}")
    
    def _plot_live_training_curves(self, history, epoch):
        """Live training curves - cập nhật mỗi epoch"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Get number of epochs from train_loss (most reliable)
        n_epochs = len(history['train_loss'])
        epochs = range(1, n_epochs + 1)
        
        # Ensure all metrics have same length
        train_loss = history['train_loss'][:n_epochs]
        val_loss = history['val_loss'][:n_epochs]
        val_rmse = history['val_rmse'][:n_epochs]
        val_pearson = history['val_pearson'][:n_epochs]
        val_ci = history['val_ci'][:n_epochs]
        
        # Train vs Val Loss
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title(f'Loss Curves (Epoch {epoch})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE
        if len(val_rmse) > 0:
            axes[0, 1].plot(epochs, val_rmse, 'g-', linewidth=2)
            best_rmse = min(val_rmse)
            best_epoch_rmse = val_rmse.index(best_rmse) + 1
            axes[0, 1].axhline(y=best_rmse, color='r', linestyle='--', 
                              label=f'Best: {best_rmse:.4f} (Epoch {best_epoch_rmse})')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pearson
        if len(val_pearson) > 0:
            axes[0, 2].plot(epochs, val_pearson, 'purple', linewidth=2)
            best_pearson = max(val_pearson)
            best_epoch_pearson = val_pearson.index(best_pearson) + 1
            axes[0, 2].axhline(y=best_pearson, color='r', linestyle='--',
                              label=f'Best: {best_pearson:.4f} (Epoch {best_epoch_pearson})')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Pearson r')
        axes[0, 2].set_title('Validation Pearson')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # CI
        if len(val_ci) > 0:
            axes[1, 0].plot(epochs, val_ci, 'orange', linewidth=2)
            best_ci = max(val_ci)
            best_epoch_ci = val_ci.index(best_ci) + 1
            axes[1, 0].axhline(y=best_ci, color='r', linestyle='--',
                              label=f'Best: {best_ci:.4f} (Epoch {best_epoch_ci})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CI')
        axes[1, 0].set_title('Concordance Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss decrease rate
        if len(train_loss) > 1:
            loss_decrease = [train_loss[i-1] - train_loss[i] 
                           for i in range(1, len(train_loss))]
            axes[1, 1].plot(range(2, len(train_loss)+1), loss_decrease, 'b-', linewidth=2)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Decrease')
            axes[1, 1].set_title('Training Speed')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Overfitting detection
        if len(train_loss) > 1:
            gap = [val_loss[i] - train_loss[i] 
                  for i in range(len(train_loss))]
            axes[1, 2].plot(epochs, gap, 'r-', linewidth=2)
            axes[1, 2].axhline(y=0, color='g', linestyle='--')
            axes[1, 2].fill_between(epochs, 0, gap, alpha=0.3)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Val Loss - Train Loss')
            axes[1, 2].set_title('Overfitting Monitor')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'live_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self, predictions, labels, epoch):
        """Scatter plot: True vs Predicted"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter
        ax.scatter(labels, predictions, alpha=0.4, s=20, c='blue', edgecolors='none')
        
        # Perfect prediction line
        min_val = min(labels.min(), predictions.min())
        max_val = max(labels.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Regression line
        z = np.polyfit(labels, predictions, 1)
        p = np.poly1d(z)
        ax.plot(labels, p(labels), 'g-', linewidth=2, alpha=0.7, 
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Metrics
        rmse = np.sqrt(np.mean((labels - predictions)**2))
        r, _ = stats.pearsonr(labels, predictions)
        
        textstr = f'Epoch {epoch}\nRMSE: {rmse:.4f}\nPearson r: {r:.4f}\nN: {len(labels)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('True Binding Affinity', fontsize=14)
        ax.set_ylabel('Predicted Binding Affinity', fontsize=14)
        ax.set_title(f'Validation Predictions (Epoch {epoch})', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'epoch_plots/scatter_epoch_{epoch:03d}.png'),
                   dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'latest_scatter.png'),
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, predictions, labels, epoch):
        """Ma trận tương quan - Correlation Matrix"""
        # Binning for correlation matrix
        n_bins = 20
        bins = np.linspace(min(labels.min(), predictions.min()),
                          max(labels.max(), predictions.max()), n_bins+1)
        
        # Create 2D histogram (correlation matrix)
        H, xedges, yedges = np.histogram2d(labels, predictions, bins=bins)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Heatmap
        im = ax.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        # Diagonal line
        ax.plot([xedges[0], xedges[-1]], [yedges[0], yedges[-1]], 
               'b--', linewidth=2, label='Perfect Prediction')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=12)
        
        # Labels
        ax.set_xlabel('True Binding Affinity', fontsize=14)
        ax.set_ylabel('Predicted Binding Affinity', fontsize=14)
        ax.set_title(f'Correlation Heatmap (Epoch {epoch})', fontsize=16, fontweight='bold')
        ax.legend()
        
        # Add metrics
        rmse = np.sqrt(np.mean((labels - predictions)**2))
        r, _ = stats.pearsonr(labels, predictions)
        textstr = f'RMSE: {rmse:.4f}\nPearson: {r:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'epoch_plots/correlation_matrix_epoch_{epoch:03d}.png'),
                   dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'latest_correlation_matrix.png'),
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_error_heatmap(self, predictions, labels, epoch):
        """Error heatmap - Phân tích lỗi theo vùng"""
        errors = np.abs(labels - predictions)
        
        # Binning
        n_bins = 15
        bins = np.linspace(min(labels.min(), predictions.min()),
                          max(labels.max(), predictions.max()), n_bins+1)
        
        # Calculate mean error for each bin
        error_matrix = np.zeros((n_bins, n_bins))
        count_matrix = np.zeros((n_bins, n_bins))
        
        for i in range(len(labels)):
            x_bin = np.digitize(labels[i], bins) - 1
            y_bin = np.digitize(predictions[i], bins) - 1
            if 0 <= x_bin < n_bins and 0 <= y_bin < n_bins:
                error_matrix[y_bin, x_bin] += errors[i]
                count_matrix[y_bin, x_bin] += 1
        
        # Average error
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_error_matrix = error_matrix / count_matrix
            avg_error_matrix = np.nan_to_num(avg_error_matrix)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Heatmap
        im = ax.imshow(avg_error_matrix, origin='lower', aspect='auto', cmap='RdYlGn_r',
                      extent=[bins[0], bins[-1], bins[0], bins[-1]])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Absolute Error', fontsize=12)
        
        # Diagonal
        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], 'b--', linewidth=2)
        
        ax.set_xlabel('True Binding Affinity', fontsize=14)
        ax.set_ylabel('Predicted Binding Affinity', fontsize=14)
        ax.set_title(f'Error Distribution Heatmap (Epoch {epoch})', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'epoch_plots/error_heatmap_epoch_{epoch:03d}.png'),
                   dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'latest_error_heatmap.png'),
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate(self, history, epoch):
        """Plot learning rate schedule"""
        # Placeholder - sẽ được update nếu có LR history
        pass
    
    def save_animation_data(self, history, epoch):
        """Lưu data để tạo animation sau này"""
        data_file = os.path.join(self.output_dir, 'animation_data.pkl')
        
        # Load existing data
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                animation_data = pickle.load(f)
        else:
            animation_data = {'epochs': [], 'history': []}
        
        # Append current epoch
        animation_data['epochs'].append(epoch)
        animation_data['history'].append({
            'train_loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1],
            'val_rmse': history['val_rmse'][-1],
            'val_pearson': history['val_pearson'][-1],
            'val_ci': history['val_ci'][-1]
        })
        
        # Save
        with open(data_file, 'wb') as f:
            pickle.dump(animation_data, f)
