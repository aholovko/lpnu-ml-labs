"""
Predict handwriting from an image using a trained model.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.lab1.model import ConvNet
from src.paths import MODELS_DIR
from src.utils import get_device, setup_logging

logger = setup_logging(logging.INFO)


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
    """Preprocess an image for model inference (for model trained on MNIST dataset)."""

    # Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize image (digit: white [255], background: black [0])
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours and select the largest one
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32).to(device)

    # Extract the digit
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = binary[y : y + h, x : x + w]

    # Deskew the digit
    m = cv2.moments(digit)
    if abs(m["mu02"]) >= 1e-2:  # Only deskew if there's significant skew
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, -skew, 0], [0, 1, 0]])
        digit = cv2.warpAffine(
            digit,
            M,
            (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0],
        )

    # Enhance thin strokes with dilation
    kernel = np.ones((2, 2), np.uint8)
    digit = cv2.dilate(digit, kernel, iterations=1)

    # Recompute bounding box after modifications
    contours, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = digit[y : y + h, x : x + w]

    # Ensure the digit has valid dimensions
    if w <= 0 or h <= 0:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32).to(device)

    # Resize to fit in 20x20 box (with 4px margin in a 28x28 image)
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in a 28x28 image
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = digit_resized

    # Convert to normalized tensor and move to device
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Apply MNIST normalization
    tensor = (tensor - 0.1307) / 0.3081

    return tensor.to(device)


def predict(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    """Make a prediction with the trained model."""

    with torch.no_grad():
        logits = model(x)
        y_pred = int(torch.argmax(logits, dim=1).item())
        prob = float(F.softmax(logits, dim=1)[0][y_pred].item())

    return y_pred, prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict handwriting from an image")
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
