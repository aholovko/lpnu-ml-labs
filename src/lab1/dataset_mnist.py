"""
Dataset module for MNIST dataset.
"""

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

from src.lab1.config import DATA_DIR, BATCH_SIZE, VALID_SIZE


class MNISTDataModule:
    """Data module for the MNIST dataset."""

    def __init__(
        self,
        data_dir: Path = Path(DATA_DIR),
        batch_size: int = BATCH_SIZE,
        valid_size: int = VALID_SIZE,
    ):
        """Initialize the MNIST data module.

        Args:
            data_dir: Directory where the dataset will be stored
            batch_size: Batch size for dataloaders
            valid_size: Number of samples to use for validation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.transform = transforms.ToTensor()
        self.train_dataset: Optional[Subset] = None
        self.valid_dataset: Optional[Subset] = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Download the MNIST dataset if not already present."""
        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self) -> None:
        """Setup train, validation, and test datasets."""
        # Load training data
        full_dataset = torchvision.datasets.MNIST(root=self.data_dir, train=True, transform=self.transform)

        # Create validation and training splits
        valid_indices = torch.arange(self.valid_size).tolist()
        train_indices = torch.arange(self.valid_size, len(full_dataset)).tolist()

        self.valid_dataset = Subset(full_dataset, valid_indices)
        self.train_dataset = Subset(full_dataset, train_indices)

        # Load test dataset
        self.test_dataset = torchvision.datasets.MNIST(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get the training dataloader.

        Args:
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader for training data
        """
        if self.train_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns:
            DataLoader for validation data
        """
        if self.valid_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.

        Returns:
            DataLoader for test data
        """
        if self.test_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
