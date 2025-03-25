"""
Utility functions.
"""

import logging
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure logging for console output."""
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

    return logging.getLogger(__name__)


# Initialize default logger
logger = setup_logging()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the appropriate PyTorch device based on availability."""
    if device_str:
        try:
            return torch.device(device_str)
        except (RuntimeError, ValueError):
            logger.warning(f"Device '{device_str}' not available, using auto-detection")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.mps, "is_available") and torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_system_info() -> Dict[str, str]:
    """Get basic system and library information."""
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"  # type: ignore

    info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda": cuda_version,
        "mps_available": str(hasattr(torch.mps, "is_available") and torch.mps.is_available()),
    }

    return info
