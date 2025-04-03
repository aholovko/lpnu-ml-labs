"""
CNN model for Audio MNIST speech recognition.
Architecture: 4 conv blocks → flatten → linear → 10 digit classes
"""

import torch
import torch.nn as nn

from src.model import BaseModel


class Net(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        # Combine all convolutional blocks into one sequential module
        self.feature_extractor = nn.Sequential(
            # Block 1: input channels=1, output channels=16
            nn.Conv2d(1, 16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Block 2: input channels=16, output channels=32
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Block 3: input channels=32, output channels=64
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Block 4: input channels=64, output channels=128
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classification head
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4480, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input spectrograms [batch_size, 1, height, width]

        Returns:
            Logits for digit classification [batch_size, num_classes]
        """
        # Extract features through convolutional layers
        features = self.feature_extractor(x)

        # Classify the features
        logits = self.classifier(features)

        return logits
