"""
Unit tests for the Trainer class.
"""

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import Trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestTrainer:
    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def loss_fn(self):
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.01)

    @pytest.fixture
    def trainer(self, model, loss_fn, optimizer):
        return Trainer(model, loss_fn, optimizer)

    @pytest.fixture
    def mock_dataloader(self):
        x = torch.randn(10, 10)  # 10 samples with 10 features
        y = torch.randint(0, 2, (10,))  # Random labels (0 or 1)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=2)

    def test_train(self, trainer, mock_dataloader):
        metrics = trainer.train(mock_dataloader, mock_dataloader, 1)

        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "valid_loss" in metrics
        assert "valid_accuracy" in metrics

        assert isinstance(metrics["train_loss"], list)
        assert len(metrics["train_loss"]) == 1
        assert isinstance(metrics["train_accuracy"], list)
        assert len(metrics["train_accuracy"]) == 1

        assert isinstance(metrics["valid_loss"], list)
        assert len(metrics["valid_loss"]) == 1
        assert isinstance(metrics["valid_accuracy"], list)
        assert len(metrics["valid_accuracy"]) == 1

    def test_evaluate(self, trainer, mock_dataloader):
        metrics = trainer.evaluate(mock_dataloader)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_save_model(self, trainer, tmp_path):
        model_path = os.path.join(tmp_path, "test_model.pt")
        trainer.save_model(path=model_path)

        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

        loaded_state = torch.load(model_path, weights_only=True)
        assert isinstance(loaded_state, dict)
