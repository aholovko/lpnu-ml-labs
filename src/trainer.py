"""
Module for training and evaluation of neural network models.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils import get_device, setup_logging

logger = setup_logging(logging.INFO)


class Trainer:
    """Trainer class for training and evaluation of neural network models."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optimizer,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The PyTorch model to train
            loss_fn: Loss function to optimize
            optimizer: Optimizer for training
            device: Device to use for training ('cpu', 'cuda', or specific device)

        Raises:
            ValueError: If any required parameter is missing
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if loss_fn is None:
            raise ValueError("Loss function cannot be None")
        if optimizer is None:
            raise ValueError("Optimizer cannot be None")

        self.device = get_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        num_epochs: int,
    ) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Args:
            train_dl: DataLoader for training data
            valid_dl: DataLoader for validation data
            num_epochs: Number of training epochs

        Returns:
            Dictionary containing training metrics history
        """
        metrics = {"train_loss": [], "valid_loss": [], "train_accuracy": [], "valid_accuracy": []}

        logger.info(f"Training on device: {self.device}")

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_dl)

            # Validation phase
            valid_loss, valid_acc = self._compute_metrics(valid_dl)

            # Save metrics
            metrics["train_loss"].append(train_loss)
            metrics["train_accuracy"].append(train_acc)
            metrics["valid_loss"].append(valid_loss)
            metrics["valid_accuracy"].append(valid_acc)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={valid_loss:.4f}, val_acc={valid_acc:.4f}"
            )

        return metrics

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Run one epoch of training.

        Args:
            dataloader: DataLoader containing batch data

        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:
            # Move data to device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.loss_fn(predictions, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate batch metrics
            batch_size = y_batch.size(0)
            running_loss += loss.item() * batch_size
            correct += (torch.argmax(predictions, dim=1) == y_batch).sum().item()
            total += batch_size

        # Calculate epoch metrics
        return running_loss / total, correct / total

    def _compute_metrics(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Compute loss and accuracy metrics on a dataset.

        Args:
            dataloader: DataLoader containing batch data

        Returns:
            Tuple of (average loss, accuracy) for the dataset
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                # Move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                predictions = self.model(x_batch)
                loss = self.loss_fn(predictions, y_batch)

                # Calculate batch metrics
                batch_size = y_batch.size(0)
                running_loss += loss.item() * batch_size
                correct += (torch.argmax(predictions, dim=1) == y_batch).sum().item()
                total += batch_size

        # Calculate metrics
        return running_loss / total, correct / total

    def evaluate(self, test_dl: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.

        Args:
            test_dl: DataLoader for test data

        Returns:
            Dictionary containing test metrics
        """
        logger.info("Evaluating on test dataset...")
        test_loss, test_acc = self._compute_metrics(test_dl)

        logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        return {"accuracy": test_acc, "loss": test_loss}

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path where to save the model
        """
        try:
            # Ensure directory exists
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    fig_size: Tuple[int, int] = (14, 5),
    dpi: int = 300,
) -> Figure:
    """
    Plot the training and validation metrics.

    Args:
        metrics: Dictionary containing training metrics history
        save_path: Optional path to save the plot image
        fig_size: Figure size as (width, height)
        dpi: Resolution of the figure

    Returns:
        Matplotlib figure object
    """
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
    plt.subplots_adjust(wspace=0.3)

    # Plot train/valid loss
    ax1.plot(epochs, metrics["train_loss"], "-o", label="Train", alpha=0.8)
    ax1.plot(epochs, metrics["valid_loss"], "--<", label="Validation", alpha=0.8)
    ax1.set_xlabel("Epoch", size=12, labelpad=10)
    ax1.set_ylabel("Loss", size=12)
    ax1.set_title("Training and Validation Loss", size=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Plot train/valid accuracy
    ax2.plot(epochs, metrics["train_accuracy"], "-o", label="Train", alpha=0.8)
    ax2.plot(epochs, metrics["valid_accuracy"], "--<", label="Validation", alpha=0.8)
    ax2.set_xlabel("Epoch", size=12, labelpad=10)
    ax2.set_ylabel("Accuracy", size=12)
    ax2.set_title("Training and Validation Accuracy", size=14)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        save_path_str = str(save_path) if isinstance(save_path, Path) else save_path
        plt.savefig(save_path_str, dpi=dpi)
        logger.info(f"Training graphs saved to {save_path}")

    return fig
