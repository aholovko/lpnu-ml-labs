"""
Utility functions.
"""

import logging
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch


def setup_logging(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and retrieve a logger with specified settings.

    Args:
        name: Logger name (typically __name__ of the calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    # Default format includes timestamp for better traceability
    if log_format is None:
        log_format = "[%(levelname)s] [%(asctime)s] %(message)s"

    # Get logger by name
    logger_name = name if name else __name__
    logger = logging.getLogger(logger_name)

    # Only configure root logger once to avoid duplicate handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler()]
        )

    # Set specific logger level
    logger.setLevel(log_level)

    return logger


# Initialize default logger
logger = setup_logging(__name__)


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
