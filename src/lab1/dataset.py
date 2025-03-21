"""
Dataset module for MNIST and EMNIST datasets.
"""

from pathlib import Path
from typing import List, Optional, Any
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.lab1.config import DATA_DIR, BATCH_SIZE, VALID_SIZE


class BaseDataModule:
    dataset_class = None

    def __init__(
        self,
        data_dir: Path = Path(DATA_DIR),
        batch_size: int = BATCH_SIZE,
        valid_size: int = VALID_SIZE,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.transform = transforms.ToTensor()
        self.train_dataset: Optional[Subset] = None
        self.valid_dataset: Optional[Subset] = None
        self.test_dataset = None

        if self.dataset_class is None:
            raise NotImplementedError("Subclasses must specify dataset_class")

    def prepare_data(self) -> None:
        if self.dataset_class is None:
            raise NotImplementedError("dataset_class cannot be None")

        self.dataset_class(root=self.data_dir, train=True, download=True)
        self.dataset_class(root=self.data_dir, train=False, download=True)

    def setup(self, **kwargs: Any) -> None:
        if self.dataset_class is None:
            raise NotImplementedError("dataset_class cannot be None")

        # Load and split training data
        full_dataset = self.dataset_class(root=self.data_dir, train=True, transform=self.transform, **kwargs)

        # Create validation and training splits
        valid_indices: List[int] = torch.arange(self.valid_size).tolist()
        train_indices: List[int] = torch.arange(self.valid_size, len(full_dataset)).tolist()

        self.valid_dataset = Subset(full_dataset, valid_indices)
        self.train_dataset = Subset(full_dataset, train_indices)

        # Load test dataset
        self.test_dataset = self.dataset_class(root=self.data_dir, train=False, transform=self.transform, **kwargs)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class MNISTDataModule(BaseDataModule):
    """Data module for the MNIST dataset."""

    dataset_class = torchvision.datasets.MNIST


class EMNISTDataModule(BaseDataModule):
    """Data module for the EMNIST dataset."""

    dataset_class = torchvision.datasets.EMNIST

    def __init__(
        self,
        split: str = "balanced",
        data_dir: Path = Path(DATA_DIR),
        batch_size: int = BATCH_SIZE,
        valid_size: int = VALID_SIZE,
    ):
        super().__init__(data_dir, batch_size, valid_size)
        self.split = split

    def prepare_data(self) -> None:
        if self.dataset_class is None:
            raise NotImplementedError("dataset_class cannot be None")

        self.dataset_class(root=self.data_dir, split=self.split, train=True, download=True)
        self.dataset_class(root=self.data_dir, split=self.split, train=False, download=True)

    def setup(self, **kwargs: Any) -> None:
        super().setup(split=self.split, **kwargs)
