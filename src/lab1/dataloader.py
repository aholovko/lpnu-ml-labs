import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from typing import List


class MNISTDataModule:
    def __init__(self, data_dir="./data", batch_size=64, valid_size=10000):
        self.test_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        """Download the MNIST dataset if not already downloaded."""
        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self):
        """Set up the train, validation, and test datasets."""
        # Load training data and split into train/validation
        mnist_full = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, transform=self.transform
        )

        valid_indices: List[int] = torch.arange(self.valid_size).tolist()
        train_indices: List[int] = torch.arange(
            self.valid_size, len(mnist_full)
        ).tolist()

        self.valid_dataset = Subset(mnist_full, valid_indices)
        self.train_dataset = Subset(mnist_full, train_indices)

        # Load test dataset
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self, shuffle=True):
        """Return the DataLoader for the training dataset."""
        if self.train_dataset is None:
            raise ValueError("Dataset not set up. Call setup() first.")

        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def val_dataloader(self):
        """Return the DataLoader for the validation dataset."""
        if self.valid_dataset is None:
            raise ValueError("Dataset not set up. Call setup() first.")

        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Return the DataLoader for the test dataset."""
        if self.test_dataset is None:
            raise ValueError("Dataset not set up. Call setup() first.")

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
