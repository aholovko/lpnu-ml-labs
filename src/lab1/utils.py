"""
Utility functions for the MNIST CNN project.
"""

import os
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)

    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (Apple Silicon)
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")

    # Fall back to CPU
    return torch.device("cpu")


def ensure_dir_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
