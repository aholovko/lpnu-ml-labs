"""
Tests for the ConvNet model architecture.
"""

import pytest
import torch
from torch import nn

from src.lab1.config import DROPOUT_RATE
from src.lab1.modeling.model import ConvNet


class TestConvNet:
    """Test suite for the ConvNet model."""

    def test_model_initialization(self):
        model = ConvNet()

        # Test model instance
        assert isinstance(model, nn.Module)

        # Test layers
        assert isinstance(model.conv1, nn.Conv2d)
        assert isinstance(model.pool, nn.MaxPool2d)
        assert isinstance(model.conv2, nn.Conv2d)
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.fc2, nn.Linear)

        # Test layer configurations
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 32
        assert model.conv1.kernel_size == (5, 5)

        assert model.conv2.in_channels == 32
        assert model.conv2.out_channels == 64
        assert model.conv2.kernel_size == (5, 5)

        assert model.fc1.in_features == 64 * 7 * 7
        assert model.fc1.out_features == 1024

        assert model.dropout.p == DROPOUT_RATE

        assert model.fc2.in_features == 1024
        assert model.fc2.out_features == 10

    def test_forward_pass(self):
        """Test that the forward pass works with correctly sized input."""
        model = ConvNet()

        # Create a batch of 5 MNIST-like images (1 channel, 28x28)
        batch_size = 5
        x = torch.randn(batch_size, 1, 28, 28)

        # Forward pass
        output = model(x)

        # Check output shape (batch_size, num_classes)
        assert output.shape == (batch_size, 10)

        # Check output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

        # Check gradients can flow (model is trainable)
        loss = output.sum()
        loss.backward()

        # Check some gradients exist
        assert model.conv1.weight.grad is not None

    def test_invalid_input_size(self):
        """Test that the model raises an error with incorrect input size."""
        model = ConvNet()

        # Input with wrong number of channels
        with pytest.raises(RuntimeError):
            x_wrong_channels = torch.randn(1, 3, 28, 28)  # 3 channels instead of 1
            model(x_wrong_channels)

        # Input with wrong spatial dimensions
        with pytest.raises(RuntimeError):
            x_wrong_size = torch.randn(1, 1, 32, 32)  # 32x32 instead of 28x28
            model(x_wrong_size)
