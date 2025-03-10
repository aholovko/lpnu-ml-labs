import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


class MNISTDataModule:
    def __init__(self, data_dir='./data', batch_size=64, valid_size=10000):
        self.test_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self):
        # Load training data and split into train/validation
        mnist_full = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform
        )

        # Split into validation and training datasets
        self.valid_dataset = Subset(mnist_full, torch.arange(self.valid_size))
        self.train_dataset = Subset(mnist_full, torch.arange(self.valid_size, len(mnist_full)))

        # Load test dataset
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform
        )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


class CNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Trainer:
    def __init__(self, model, loss_fn=None, optimizer=None, device=None):
        # Set device (GPU if available, otherwise CPU)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Default loss function and optimizer if not provided
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.001)

    def train(self, num_epochs, train_dl, valid_dl):
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
                f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')

        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

    def evaluate(self, test_dataset):
        self.model.eval()
        self.model = self.model.cpu()  # Move model to CPU for inference

        # Get predictions for the entire test dataset
        with torch.no_grad():
            pred = self.model(test_dataset.data.unsqueeze(1) / 255.)
            is_correct = (torch.argmax(pred, dim=1) == test_dataset.targets).float()
            test_accuracy = is_correct.mean().item()

        print(f'Test accuracy: {test_accuracy:.4f}')
        return test_accuracy

    def save_model(self, path='models/mnist-cnn.pt'):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        torch.save(self.model, path)
        print(f'Model saved to {path}')


def plot_training_history(hist, save_path=None):
    x_arr = np.arange(len(hist[0])) + 1

    fig = plt.figure(figsize=(12, 4))

    # Loss subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label='Train loss')
    ax.plot(x_arr, hist[1], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)

    # Accuracy subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label='Train acc.')
    ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Initialize the data module
    data_module = MNISTDataModule(batch_size=64)
    data_module.prepare_data()
    data_module.setup()

    # Create data loaders
    train_dl = data_module.train_dataloader()
    valid_dl = data_module.val_dataloader()

    # Create the model
    model = CNN(dropout_rate=0.5)

    # Initialize trainer
    trainer = Trainer(model)

    # Train the model
    num_epochs = 5
    hist = trainer.train(num_epochs, train_dl, valid_dl)

    # Plot training history
    plot_training_history(hist)

    # Evaluate on test set
    test_accuracy = trainer.evaluate(data_module.test_dataset)

    # Save the model
    trainer.save_model()

    return model, hist, test_accuracy


if __name__ == "__main__":
    main()
