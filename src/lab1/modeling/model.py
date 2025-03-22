"""
Module containing model architecture for MNIST classification.
"""

import torch.nn as nn
import torch.nn.functional as F

from src.lab1.config import DROPOUT_RATE


class ConvNet(nn.Module):
    """
    CNN model that consists of two convolutional blocks followed by fully connected layers.
    """

    def __init__(self):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  # 64 channels after 2 pooling layers
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))  # Use functional ReLU
        x = self.pool(x)

        # Second block
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # Flatten with explicit dimensions
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
