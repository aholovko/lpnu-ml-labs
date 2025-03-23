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


def predict(model, image_tensor) -> Tuple[int, float]:
    """Make a prediction with the trained model."""

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = int(torch.argmax(outputs, dim=1).item())
        confidence = float(F.softmax(outputs, dim=1)[0][predicted_class].item())

    return predicted_class, confidence


def load_model(model_path: Path, device: torch.device):
    """Load a trained model from disk."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ConvNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    return model


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load and preprocess an image for model inference."""

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        raise

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
    image = cv2.bitwise_not(image)
    image = image.astype(np.float32) / 255.0

    tensor_image = torch.from_numpy(image).float()

    return tensor_image.unsqueeze(0).to(device)


def main():
    """Run handwriting prediction."""

    parser = argparse.ArgumentParser(description="Predict handwriting")
    parser.add_argument("--model-name", required=True, help="model name to load")
    parser.add_argument("--image-path", required=True, help="path to the image")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    args = parser.parse_args()

    device = get_device(args.device)
    model_path = Path(MODELS_DIR) / f"{args.model_name}.pt"

    model = load_model(model_path, device)
    image_tensor = load_image(Path(args.image_path), device)
    predicted_class, confidence = predict(model, image_tensor)

    logger.info(f'"{predicted_class}" (confidence: {confidence * 100:.1f}%)')


if __name__ == "__main__":
    main()
