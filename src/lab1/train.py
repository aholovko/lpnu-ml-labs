"""
Train and evaluate a model for handwriting recognition.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import torch

from src.lab1.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    VALID_SIZE,
)
from src.lab1.dataset_mnist import MNISTDataModule
from src.lab1.model import ConvNet
from src.paths import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.trainer import Trainer, plot_training_metrics
from src.utils import get_system_info, set_seed, setup_logging

logger = setup_logging(logging.INFO)


def main():
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, loss_fn, optimizer, args.device)

    # Generate model name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mnist_classifier_{timestamp}"

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    metrics = trainer.train(data_module.train_dataloader(), data_module.val_dataloader(), args.epochs)

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    trainer.save_model(path=model_path)

    # Plot loss and accuracy graphs
    plot_path = os.path.join(FIGURES_DIR, f"training_{model_name}.png")
    plot_training_metrics(metrics, save_path=plot_path)

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
        "system_info": get_system_info(),
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
