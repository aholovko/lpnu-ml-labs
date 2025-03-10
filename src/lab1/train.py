import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

from src.lab1.dataloader import MNISTDataModule
from src.lab1.model import CNN


class Trainer:
    def __init__(self, model, loss_fn=None, optimizer=None, device=None):
        # Set device (GPU if available, otherwise CPU)
        self.device = (
            device
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        # Default loss function and optimizer if not provided
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.001)
        )

    def train(
        self, num_epochs, train_dl, valid_dl
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for
            train_dl: Training data loader
            valid_dl: Validation data loader

        Returns:
            A tuple containing (train_loss_history, val_loss_history, train_acc_history, val_acc_history)
        """
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            for x_batch, y_batch in train_dl:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Accumulate metrics
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum().cpu()

            # Calculate average training metrics
            loss_hist_train[epoch] /= len(train_dl.dataset)
            accuracy_hist_train[epoch] /= len(train_dl.dataset)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    pred = self.model(x_batch)
                    loss = self.loss_fn(pred, y_batch)

                    # Accumulate metrics
                    loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                    is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                    accuracy_hist_valid[epoch] += is_correct.sum().cpu()

            # Calculate average validation metrics
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

            print(
                f"Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}"
            )

        return (
            loss_hist_train,
            loss_hist_valid,
            accuracy_hist_train,
            accuracy_hist_valid,
        )

    def save_model(self, path="models/mnist-cnn.pt"):
        """Save the trained model to the specified path."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        torch.save(self.model, path)
        print(f"Model saved to {path}")


def plot_training_history(hist, save_path=None):
    """
    Plot the training history.

    Args:
        hist: A tuple containing (train_loss_history, val_loss_history, train_acc_history, val_acc_history)
        save_path: Optional path to save the plot to
    """
    x_arr = np.arange(len(hist[0])) + 1

    fig = plt.figure(figsize=(12, 4))

    # Loss subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], "-o", label="Train loss")
    ax.plot(x_arr, hist[1], "--<", label="Validation loss")
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Loss", size=15)
    ax.legend(fontsize=15)

    # Accuracy subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], "-o", label="Train acc.")
    ax.plot(x_arr, hist[3], "--<", label="Validation acc.")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Accuracy", size=15)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")

    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a CNN model on MNIST dataset")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate (default: 0.5)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="directory to store the dataset (default: ./data)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mnist-cnn.pt",
        help="path to save the model (default: models/mnist-cnn.pt)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="training_history.png",
        help="path to save the training history plot (default: training_history.png)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=10000,
        help="size of validation dataset (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize the data module
    data_module = MNISTDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size, valid_size=args.valid_size
    )
    data_module.prepare_data()
    data_module.setup()

    # Create data loaders
    train_dl = data_module.train_dataloader()
    valid_dl = data_module.val_dataloader()

    # Create the model
    model = CNN(dropout_rate=args.dropout)

    # Initialize trainer with custom learning rate if provided
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer=optimizer)

    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    hist = trainer.train(args.epochs, train_dl, valid_dl)

    # Plot training history
    plot_training_history(hist, save_path=args.plot_path)

    # Save the model
    trainer.save_model(path=args.model_path)

    print("Training completed successfully!")
