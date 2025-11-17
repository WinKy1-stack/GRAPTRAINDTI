"""
Utility functions for setting random seeds
Ensures reproducibility across PyTorch, NumPy, and Python random
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def get_device(prefer_cuda: bool = True):
    """
    Get available device (CUDA / CPU)
    
    Args:
        prefer_cuda (bool): Try to use CUDA if available
    
    Returns:
        torch.device: Device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device
