import os
import torch
from torch import Tensor
import argparse
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.lab1.features import extract_features
from src.lab1.utils import get_device
from src.lab1.config import MODELS_DIR


def load_image(image_path) -> Tensor:
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Adjust image if needed
    if max(image.size) > 100:
        # Increase contrast
        image = ImageOps.autocontrast(image, cutoff=5)

        # Check if image needs inversion (MNIST has white digits on black)
        pixel_data = np.array(image)
        avg_pixel = np.mean(pixel_data)
        if avg_pixel > 128:
            image = ImageOps.invert(image)

        # Crop to content
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

    # Resize and convert to tensor
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    return transform(image)  # type: ignore # ToTensor() returns a torch.Tensor


def predict(model_path, image_path, device=None):
    if device is None:
        device = get_device()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model {model_path} not found")

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")

    image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(device)

    features = extract_features(image_tensor)

    if len(features.shape) == 3:
        features = features.unsqueeze(0)

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_idx = int(predicted_class)
        confidence = probabilities[0][predicted_idx].item()

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    if len(image_tensor.shape) == 4:
        img_display = image_tensor[0, 0].cpu().numpy()
    else:
        img_display = image_tensor[0].cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img_display, cmap="gray")
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict digits from images")
    parser.add_argument(
        "--model-name", required=True, help="model filename (without extension)"
    )
    parser.add_argument("--image-path", required=True, help="path to the image")
    parser.add_argument("--device", default=None, help="device for inference")
    args = parser.parse_args()

    model_path = Path(MODELS_DIR) / f"{args.model_name}.pt"

    predicted_class, confidence = predict(model_path, args.image_path, args.device)

    return predicted_class, confidence


if __name__ == "__main__":
    main()
