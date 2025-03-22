"""
Module for training and evaluating CNN models on MNIST data.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn

from src.lab1.config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    VALID_SIZE,
)
from src.lab1.dataset_mnist import MNISTDataModule
from src.lab1.modeling.model import ConvNet
from src.lab1.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training and evaluation of model."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn=None,
        optimizer=None,
        device=None,
    ):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train(self, num_epochs, train_dl, valid_dl):
        """Train the model for the specified number of epochs."""

        metrics = {"train_loss": [], "valid_loss": [], "train_accuracy": [], "valid_accuracy": []}

        logger.info(f"Using device: {self.device}")

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss, train_acc = self._run_epoch(train_dl, is_training=True)

            # Validation
            self.model.eval()
            valid_loss, valid_acc = self._run_epoch(valid_dl, is_training=False)

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

    def _run_epoch(self, dataloader, is_training=True):
        """Run one epoch of training or validation."""

        total_loss, correct, total = 0, 0, 0

        with torch.set_grad_enabled(is_training):
            for x_batch, y_batch in dataloader:
                # Move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                predictions = self.model(x_batch)
                loss = self.loss_fn(predictions, y_batch)

                # Backward pass (training only)
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Calculate batch metrics
                batch_size = y_batch.size(0)
                total_loss += loss.item() * batch_size
                correct += (torch.argmax(predictions, dim=1) == y_batch).sum().item()
                total += batch_size

        # Calculate epoch metrics
        return total_loss / total, correct / total

    def evaluate(self, test_dl):
        """Evaluate the model on a test dataset."""

        logger.info("Evaluating on test dataset...")
        self.model.eval()
        test_loss, test_acc = self._run_epoch(test_dl, is_training=False)

        logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        return {"accuracy": test_acc, "loss": test_loss}

    def save_model(self, path):
        """Save the trained model to disk."""

        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")


def plot_training_metrics(metrics, save_path=None):
    """Plot the training and validation metrics."""

    epochs = np.arange(1, len(metrics["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Plot train/valid loss
    ax1.plot(epochs, metrics["train_loss"], "-o", label="Train")
    ax1.plot(epochs, metrics["valid_loss"], "--<", label="Validation")
    ax1.set_xlabel("Epoch", size=15, labelpad=10)
    ax1.set_ylabel("Loss", size=15)
    ax1.legend(fontsize=15)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Plot train/valid accuracy
    ax2.plot(epochs, metrics["train_accuracy"], "-o", label="Train")
    ax2.plot(epochs, metrics["valid_accuracy"], "--<", label="Validation")
    ax2.set_xlabel("Epoch", size=15, labelpad=10)
    ax2.set_ylabel("Accuracy", size=15)
    ax2.legend(fontsize=15)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Training graphs saved to {save_path}")


def main():
    """Training and evaluation model."""

    parser = argparse.ArgumentParser(description="Train and evaluate model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"number of epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))
    set_seed(SEED)

    # Prepare data
    data_module = MNISTDataModule(Path(DATA_DIR), BATCH_SIZE, VALID_SIZE)
    data_module.prepare_data()
    data_module.setup()

    # Create model and trainer
    model = ConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer=optimizer, device=args.device)

    # Generate model name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mnist_classifier_{timestamp}"

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(args.epochs, data_module.train_dataloader(), data_module.val_dataloader())

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    trainer.save_model(path=model_path)

    # Plot loss and accuracy graphs
    plot_path = os.path.join(FIGURES_DIR, f"training_{model_name}.png")
    plot_training_metrics(history, save_path=plot_path)

    # Evaluate model on test dataset
    test_metrics = trainer.evaluate(data_module.test_dataloader())

    # Prepare report
    dataset_sizes = {
        "train": len(data_module.train_dataset) if data_module.train_dataset is not None else 0,
        "validation": len(data_module.valid_dataset) if data_module.valid_dataset is not None else 0,
        "test": len(data_module.test_dataset) if data_module.test_dataset is not None else 0,
    }

    report = {
        "model_name": model_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(trainer.device),
        "dataset_sizes": dataset_sizes,
        "training_parameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": BATCH_SIZE,
        },
        "test_metrics": test_metrics,
    }

    # Save report
    json_path = os.path.join(REPORTS_DIR, f"report_{model_name}.json")
    try:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")


if __name__ == "__main__":
    main()
