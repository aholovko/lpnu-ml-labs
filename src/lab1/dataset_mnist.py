"""
Data module for the MNIST dataset.
"""

from pathlib import Path
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.datamodule import BaseDataModule
from src.lab1.config import BATCH_SIZE, VALID_SIZE
from src.paths import DATA_DIR

# MNIST normalization parameters
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTDataModule(BaseDataModule):
    """Data module for the MNIST dataset."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        valid_size: int = VALID_SIZE,
    ):
        """Initialize the MNIST data module.

        Args:
            data_dir: Directory where the dataset will be stored
            batch_size: Batch size for dataloaders
            valid_size: Number of samples to use for validation
        """
        BaseDataModule.__init__(self, data_dir, batch_size)
        self.valid_size = valid_size
        self.transform = self._get_transforms()
        self.train_dataset: Optional[Subset] = None
        self.valid_dataset: Optional[Subset] = None
        self.test_dataset = None

    @staticmethod
    def _get_transforms():
        """Get MNIST-specific transforms."""
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD])])

    def prepare_data(self) -> None:
        """Download the MNIST dataset if not already present."""
        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self) -> None:
        """Setup train, validation, and test datasets."""
        # Load training data
        dataset = torchvision.datasets.MNIST(root=self.data_dir, train=True, transform=self.transform)

        # Create validation and training splits
        valid_indices = torch.arange(self.valid_size).tolist()
        train_indices = torch.arange(self.valid_size, len(dataset)).tolist()

        self.valid_dataset = Subset(dataset, valid_indices)
        self.train_dataset = Subset(dataset, train_indices)

        # Load test dataset
        self.test_dataset = torchvision.datasets.MNIST(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        if self.valid_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
