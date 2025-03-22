"""
Unit tests for the MNIST DataModule.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from src.lab1.config import BATCH_SIZE, DATA_DIR, VALID_SIZE
from src.lab1.dataset_mnist import MNISTDataModule


@pytest.fixture(scope="module")
def data_module():
    """Fixture providing a prepared MNIST DataModule."""
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()
    return dm


def test_initialization():
    """Test DataModule initialization with default parameters."""
    dm = MNISTDataModule()

    assert dm.data_dir == Path(DATA_DIR)
    assert dm.batch_size == BATCH_SIZE
    assert dm.valid_size == VALID_SIZE
    assert all(dataset is None for dataset in [dm.train_dataset, dm.valid_dataset, dm.test_dataset])


def test_datasets(data_module):
    """Test that setup creates datasets with the expected properties."""
    # Check datasets were created
    assert all(
        dataset is not None
        for dataset in [data_module.train_dataset, data_module.valid_dataset, data_module.test_dataset]
    )

    # Check dataset sizes
    assert len(data_module.valid_dataset) == data_module.valid_size
    assert len(data_module.train_dataset) == 60000 - data_module.valid_size


@pytest.mark.parametrize(
    "loader_func,shuffle_expected", [("train_dataloader", True), ("val_dataloader", False), ("test_dataloader", False)]
)
def test_dataloaders(data_module, loader_func, shuffle_expected):
    """Test dataloader creation and properties."""
    # Get the dataloader
    loader = getattr(data_module, loader_func)()

    # Check dataloader properties
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == data_module.batch_size
    assert isinstance(loader.sampler, RandomSampler) == shuffle_expected


def test_train_dataloader_no_shuffle(data_module):
    """Test that train_dataloader respects shuffle=False parameter."""
    loader = data_module.train_dataloader(shuffle=False)
    assert not isinstance(loader.sampler, RandomSampler)


def test_dataloader_content(data_module):
    """Test that dataloaders return correctly formatted data."""
    # Get a batch from the train dataloader
    images, labels = next(iter(data_module.train_dataloader()))

    # Check shapes and types
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (data_module.batch_size, 1, 28, 28)
    assert labels.shape == (data_module.batch_size,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_error_handling():
    """Test that accessing dataloaders without setup raises appropriate errors."""
    dm = MNISTDataModule()

    for loader_func in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        with pytest.raises(ValueError, match="Call setup()"):
            getattr(dm, loader_func)()
