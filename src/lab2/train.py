"""
Train and evaluate a model for speech recognition.
"""

import logging
from argparse import ArgumentParser

from src.lab2.config import SEED
from src.lab2.dataset_audio_mnist import AudioMNISTDataset
from src.utils import set_seed, setup_logging

logger = setup_logging(logging.INFO)


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

    logger.setLevel(getattr(logging, args.log_level))
    set_seed(SEED)

    # Prepare data
    _ = AudioMNISTDataset(download=True)


if __name__ == "__main__":
    main()
