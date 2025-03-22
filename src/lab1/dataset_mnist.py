"""
Data module for MNIST dataset.
"""

from pathlib import Path
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.lab1.config import BATCH_SIZE, DATA_DIR, VALID_SIZE


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
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
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
