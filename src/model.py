"""
Base class for models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch

from src.utils import setup_logging

logger = setup_logging()


class BaseModel(ABC, torch.nn.Module):
    """Abstract base class for models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def load(self, path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        """Load model weights from a file."""
        try:
            load_path = Path(path) if isinstance(path, str) else path
            if device is None:
                device = torch.device("cpu")

            self.load_state_dict(torch.load(load_path, map_location=device))
            logger.info(f"Loaded model from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {str(e)}")

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights to a file."""
        try:
            save_path = Path(path) if isinstance(path, str) else path
            torch.save(self.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
