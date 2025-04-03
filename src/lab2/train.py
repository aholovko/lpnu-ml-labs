"""
Train and evaluate a model for speech recognition.
"""

import datetime
import logging
from argparse import ArgumentParser

import torch
import torchaudio.transforms as T

from src.lab2.config import SAMPLE_RATE, SEED
from src.lab2.dataset_audio_mnist import AudioMNISTDataModule
from src.lab2.model import Net
from src.paths import (
    FIGURES_DIR,
    MODELS_DIR,
)
from src.trainer import Trainer, plot_training_metrics
from src.utils import set_seed, setup_logging

logger = setup_logging(__name__)


def main() -> None:
    parser = ArgumentParser(description="Train and evaluate model")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Reconfigure logger with specified log level
    global logger
    logger = setup_logging(__name__, getattr(logging, args.log_level))
    set_seed(SEED)

    # Prepare data
    transform = torch.nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64),
        T.AmplitudeToDB(stype="power", top_db=80),
    )

    data_module = AudioMNISTDataModule(
        valid_size=4500,
        test_size=4500,
        download=False,
        transform=transform,
    )
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model and trainer
    model = Net()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, loss_fn, optimizer, args.device)

    # Generate model name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"audio_mnist_classifier_{timestamp}"

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    metrics = trainer.train(train_loader, val_loader, args.epochs)

    # Save model
    model_path = MODELS_DIR / f"{model_name}.pt"
    trainer.save_model(path=str(model_path))

    # Plot loss and accuracy graphs
    plot_path = FIGURES_DIR / f"training_{model_name}.png"
    plot_training_metrics(metrics, save_path=str(plot_path))


if __name__ == "__main__":
    main()
