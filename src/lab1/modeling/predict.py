"""
Module for handwriting recognition using trained model.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps

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
        image = ImageOps.grayscale(Image.open(image_path))
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        raise

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    tensor_image: torch.Tensor = torch.as_tensor(transform(image))
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
