"""
Evaluation metrics for Drug-Target Interaction prediction
- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error)
- Pearson Correlation Coefficient
- Spearman Correlation Coefficient
- Concordance Index (CI)
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        float: MSE value
    """
    return mean_squared_error(y_true, y_pred)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Pearson Correlation Coefficient
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        Tuple[float, float]: (correlation, p-value)
    """
    if len(y_true) < 2:
        return 0.0, 1.0
    return stats.pearsonr(y_true, y_pred)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Spearman Correlation Coefficient
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        Tuple[float, float]: (correlation, p-value)
    """
    if len(y_true) < 2:
        return 0.0, 1.0
    return stats.spearmanr(y_true, y_pred)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Index (C-Index)
    Measures ranking quality - higher is better
    
    Reference: 
    Harrell, F. E., et al. (1982). "Evaluating the yield of medical tests."
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        float: CI value (0.5 = random, 1.0 = perfect)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n = len(y_true)
    if n < 2:
        return 0.5
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    ties = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] < y_pred[j]:
                    discordant += 1
                else:
                    ties += 1
            elif y_true[i] < y_true[j]:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] > y_pred[j]:
                    discordant += 1
                else:
                    ties += 1
    
    total_pairs = concordant + discordant + ties
    if total_pairs == 0:
        return 0.5
    
    ci = (concordant + 0.5 * ties) / total_pairs
    return ci


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all metrics at once
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
    
    Returns:
        dict: Dictionary of metric names and values
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    pearson_r, pearson_p = pearson(y_true, y_pred)
    spearman_r, spearman_p = spearman(y_true, y_pred)
    
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'pearson': pearson_r,
        'pearson_p': pearson_p,
        'spearman': spearman_r,
        'spearman_p': spearman_p,
        'ci': concordance_index(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Print metrics in a formatted way
    
    Args:
        metrics (dict): Dictionary of metrics
        prefix (str): Prefix for print (e.g., "Train", "Val", "Test")
    """
    print(f"\n{'='*50}")
    print(f"{prefix} Metrics:")
    print(f"{'='*50}")
    print(f"RMSE:            {metrics['rmse']:.4f}")
    print(f"MSE:             {metrics['mse']:.4f}")
    print(f"Pearson r:       {metrics['pearson']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"Spearman rho:    {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"Concordance Index: {metrics['ci']:.4f}")
    print(f"{'='*50}\n")
