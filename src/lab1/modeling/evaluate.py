"""
Module for evaluating trained models on the MNIST test dataset.
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from src.lab1.dataset import MNISTDataModule
from src.lab1.features import extract_features
from src.lab1.utils import get_device, set_seed
from src.lab1.config import MODELS_DIR, DATA_DIR, BATCH_SIZE, SEED, REPORTS_DIR


class ModelEvaluator:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.model_path = Path(MODELS_DIR) / f"{model_name}.pt"
        self.device = get_device(device)
        self.model = None
        self.data_module = MNISTDataModule(
            data_dir=Path(DATA_DIR), batch_size=BATCH_SIZE
        )

        reports_dir = Path(REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = reports_dir / f"eval_{model_name}.txt"

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model '{self.model_name}' not found at {self.model_path}"
            )

        self.model = torch.load(self.model_path)
        print(f"Loaded model '{self.model_name}'")

    def evaluate(self) -> float:
        if self.model is None:
            self.load_model()

        if self.model is None:
            raise RuntimeError("Failed to load model")

        self.model.to(self.device)
        self.model.eval()

        test_loader = self.data_module.test_dataloader()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = extract_features(images)

                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")

        return accuracy

    def save_results(self, accuracy: float) -> None:
        with open(self.output_file, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")

        print(f"Results saved to {self.output_file}")

    def run(self) -> float:
        print(f"Device: {self.device}")

        self.data_module.prepare_data()
        self.data_module.setup()

        print(f"Evaluating model '{self.model_name}'...")
        accuracy = self.evaluate()
        self.save_results(accuracy)

        return accuracy


def parse_args():
    parser = ArgumentParser(description="Evaluate a trained model on MNIST")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="name of the model to evaluate (without extension)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use (default: auto)"
    )

    return parser.parse_args()


def main() -> float:
    args = parse_args()
    set_seed(SEED)

    evaluator = ModelEvaluator(model_name=args.model_name, device=args.device)

    return evaluator.run()


if __name__ == "__main__":
    main()
