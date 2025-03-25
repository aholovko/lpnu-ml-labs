"""
Utility functions for Lab1 project.
"""

import logging
import random
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the appropriate PyTorch device."""

    if device_str:
        try:
            return torch.device(device_str)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Requested device '{device_str}' not available: {str(e)}")
            logger.warning("Falling back to auto-detection")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
