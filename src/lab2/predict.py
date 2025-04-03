"""
Predict spoken digit from an audio file using a trained model.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from src.lab2.config import SAMPLE_RATE
from src.lab2.model import Net
from src.paths import MODELS_DIR
from src.utils import get_device, setup_logging

logger = setup_logging(__name__)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    return model


def load_audio(audio_path: str) -> torch.Tensor:
    """Load an audio file from disk."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
        if waveform is None:
            raise FileNotFoundError(f"Audio file not found or could not be read: {audio_path}")

        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        raise


def preprocess_audio(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Preprocess an audio file for model inference."""

    # Ensure audio has correct shape (mono)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Trim or pad to SAMPLE_RATE samples
    if waveform.shape[1] > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
    elif waveform.shape[1] < SAMPLE_RATE:
        padding = SAMPLE_RATE - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))

    # Apply the same transformations used in training
    transform = torch.nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64),
        T.AmplitudeToDB(stype="power", top_db=80),
    )

    # Apply transformation and add batch dimension
    mel_spectrogram = transform(waveform)

    # Add batch dimension if not present
    if mel_spectrogram.dim() == 3:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)

    return mel_spectrogram.to(device)


def predict(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    with torch.no_grad():
        logits = model(x)
        y_pred = int(torch.argmax(logits, dim=1).item())
        prob = float(F.softmax(logits, dim=1)[0][y_pred].item())

    return y_pred, prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spoken digit from an audio file")
    parser.add_argument("--model-name", required=True, help="model name to load")
    parser.add_argument("--audio-path", required=True, help="path to the audio file")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: auto)")
    args = parser.parse_args()

    device = get_device(args.device)
    model_path = MODELS_DIR / f"{args.model_name}.pt"
    model = load_model(str(model_path), device)

    waveform = load_audio(args.audio_path)
    x = preprocess_audio(waveform, device)

    y_pred, prob = predict(model, x)
    logger.info(f'Predicted digit: "{y_pred}" (probability: {prob * 100:.1f}%)')


if __name__ == "__main__":
    main()
