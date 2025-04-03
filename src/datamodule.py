"""
Base class for data modules.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader


class BaseDataModule(ABC):
    """Abstract base class for data modules."""

    def __init__(self, data_dir: Path, batch_size: int):
        """
        Initialize the data module.

        Args:
            data_dir: Directory where the dataset will be stored
            batch_size: Batch size for dataloaders
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

    @abstractmethod
    def prepare_data(self) -> None:
        """Download or prepare the dataset if not already done."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Setup train, validation, and test datasets."""
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        pass
