"""
Module for training the CNN model on MNIST data.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import datetime
from typing import Tuple, List
from pathlib import Path

from src.lab1.dataset import MNISTDataModule
from src.lab1.modeling.model import ConvNet
from src.lab1.features import extract_features
from src.lab1.utils import get_device, set_seed
from src.lab1.config import (
    DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    SEED,
    VALID_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
)


class Trainer:
    def __init__(self, model, loss_fn=None, optimizer=None, device=None):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train(self, num_epochs, train_dl, valid_dl) -> Tuple[List[float], List[float], List[float], List[float]]:
        print(f"Using device: {self.device}")

        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss, train_acc = self._run_epoch(train_dl, is_training=True)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            self.model.eval()
            valid_loss, valid_acc = self._run_epoch(valid_dl, is_training=False)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
                f"val_loss={valid_loss:.4f}, val_acc={valid_acc:.4f}"
            )

        return train_losses, valid_losses, train_accs, valid_accs

    def _run_epoch(self, dataloader, is_training=True):
        total_loss, correct, total = 0, 0, 0

        torch.set_grad_enabled(is_training)
        for x_batch, y_batch in dataloader:
            x_batch = extract_features(x_batch.to(self.device))
            y_batch = y_batch.to(self.device)

            # Forward pass
            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)

            # Backward pass (training only)
            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item() * y_batch.size(0)
            correct += (torch.argmax(pred, dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        torch.set_grad_enabled(True)
        return total_loss / total, correct / total

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        print(f"Model saved to {path}")


def plot_training_history(hist, save_path=None):
    train_losses, val_losses, train_accs, val_accs = hist
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Loss plot
    ax1.plot(epochs, train_losses, "-o", label="Train")
    ax1.plot(epochs, val_losses, "--<", label="Validation")
    ax1.set_xlabel("Epoch", size=15, labelpad=10)
    ax1.set_ylabel("Loss", size=15)
    ax1.legend(fontsize=15)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Accuracy plot
    ax2.plot(epochs, train_accs, "-o", label="Train")
    ax2.plot(epochs, val_accs, "--<", label="Validation")
    ax2.set_xlabel("Epoch", size=15, labelpad=10)
    ax2.set_ylabel("Accuracy", size=15)
    ax2.legend(fontsize=15)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"number of epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    data_module = MNISTDataModule(Path(DATA_DIR), BATCH_SIZE, VALID_SIZE)
    data_module.prepare_data()
    data_module.setup()

    model = ConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer=optimizer, device=args.device)

    print(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(args.epochs, data_module.train_dataloader(), data_module.val_dataloader())

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"mnist_classifier_{timestamp}.pt")
    plot_path = os.path.join(FIGURES_DIR, f"training_{timestamp}.png")

    plot_training_history(history, save_path=plot_path)
    trainer.save_model(path=model_path)

    print("Training completed!")


if __name__ == "__main__":
    main()
