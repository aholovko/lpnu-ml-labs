"""
Module for handwriting recognition using trained model.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.lab1.config import MODELS_DIR
from src.lab1.modeling.model import ConvNet
from src.lab1.utils import get_device

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from disk."""

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ConvNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    return model


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk."""

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess an image for model inference."""

    image_size = (28, 28)
    threshold_block_size = 11
    threshold_c = 7

    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LANCZOS4)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_block_size, threshold_c
    )
    image = cv2.bitwise_not(image)
    image = image.astype(np.float32) / 255.0

    x_tensor = torch.from_numpy(image).float()
    return x_tensor.unsqueeze(0).to(device)


def predict(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    """Make a prediction with the trained model."""

    with torch.no_grad():
        logits = model(x)
        y_pred = int(torch.argmax(logits, dim=1).item())
        prob = float(F.softmax(logits, dim=1)[0][y_pred].item())

    return y_pred, prob


def main() -> None:
    """Run handwriting prediction."""

    parser = argparse.ArgumentParser(description="Predict handwriting")
    parser.add_argument("--model-name", required=True, help="model name to load")
    parser.add_argument("--image-path", required=True, help="path to the image")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    args = parser.parse_args()

    device = get_device(args.device)
    model_path = str(Path(MODELS_DIR) / f"{args.model_name}.pt")
    model = load_model(model_path, device)

    image = load_image(args.image_path)
    x = preprocess_image(image, device)

    y_pred, prob = predict(model, x)
    logger.info(f'Predicted: "{y_pred}" (probability: {prob * 100:.1f}%)')


if __name__ == "__main__":
    main()
