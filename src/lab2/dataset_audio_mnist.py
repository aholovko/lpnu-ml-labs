"""
AudioMNIST Dataset

A spoken digits dataset consisting of 30,000 audio samples of digits (0-9) recorded by
60 different speakers (12 female/48 male). Each speaker contributed 50 recordings of
each digit, resulting in a balanced dataset (500 recordings per speaker).

Dataset Structure:
- 10 digits (0-9)
- 60 speakers (labeled 01-60)
- 50 recordings per digit per speaker (labeled 0-49)
- Audio format: WAV files (48 kHz, mono, 16-bit)

File Naming Convention:
- Files are stored in folders named with zero-padded speaker id (e.g., "01" for speaker 1)
- Filename format: {digit}_{speaker_id_padded}_{repetition_id}.wav
- Example: 0_01_0.wav (first recording [0] of digit "0" by speaker "01")

Citation:
@article{audiomnist2023,
    title = {AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark},
    journal = {Journal of the Franklin Institute},
    year = {2023},
    issn = {0016-0032},
    doi = {https://doi.org/10.1016/j.jfranklin.2023.11.038},
    url = {https://www.sciencedirect.com/science/article/pii/S0016003223007536},
    author = {Sören Becker and Johanna Vielhaben and Marcel Ackermann and Klaus-Robert Müller and Sebastian Lapuschkin and Wojciech Samek},
    keywords = {Deep learning, Neural networks, Interpretability, Explainable artificial intelligence, Audio classification, Speech recognition},
}

Resources:
- GitHub repo: https://github.com/soerenab/AudioMNIST
- Metadata file: https://raw.githubusercontent.com/soerenab/AudioMNIST/refs/heads/master/data/audioMNIST_meta.txt
- Download URL format: https://raw.githubusercontent.com/soerenab/AudioMNIST/refs/heads/master/data/{speaker_id_padded}/{digit}_{speaker_id_padded}_{repetition_id}.wav
- Example: https://raw.githubusercontent.com/soerenab/AudioMNIST/refs/heads/master/data/01/0_01_0.wav
"""  # noqa: E501

import logging
import os
import shutil
import zipfile
from typing import Tuple
from urllib import request

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.lab2.config import BATCH_SIZE, SAMPLE_RATE
from src.paths import DATA_DIR
from src.utils import setup_logging

logger = setup_logging(logging.INFO)


AUDIO_MNIST_ZIP_URL = "https://github.com/soerenab/AudioMNIST/archive/refs/heads/master.zip"
AUDIO_MNIST_META_FILE = "audioMNIST_meta.txt"
AUDIO_MNIST_DIR = os.path.join(DATA_DIR, "AUDIOMNIST")
AUDIO_MNIST_RAW_DIR = os.path.join(AUDIO_MNIST_DIR, "raw")


class AudioMNISTDataset(Dataset):
    """AudioMNIST dataset."""

    def __init__(self, download: bool = True, num_samples: int = SAMPLE_RATE):
        self.num_samples = num_samples

        if download:
            self._download_dataset_if_needed()

        self._load_data()

    @staticmethod
    def _download_dataset_if_needed() -> bool:
        meta_file_path = os.path.join(AUDIO_MNIST_DIR, AUDIO_MNIST_META_FILE)

        if os.path.exists(meta_file_path):
            return True

        os.makedirs(AUDIO_MNIST_DIR, exist_ok=True)

        zip_path = os.path.join(AUDIO_MNIST_DIR, "audiomnist.zip")
        logger.info(f"Downloading AudioMNIST archive from {AUDIO_MNIST_ZIP_URL}")

        with tqdm(unit="B", unit_scale=True, miniters=1, desc="AudioMNIST") as progress_bar:

            def update_progress(count, block_size, total_size):
                if total_size is not None:
                    progress_bar.total = total_size
                progress_bar.update(block_size)

            request.urlretrieve(AUDIO_MNIST_ZIP_URL, filename=zip_path, reporthook=update_progress)

        logger.info(f"Extracting archive to {AUDIO_MNIST_DIR}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(AUDIO_MNIST_DIR)

        extracted_dir = os.path.join(AUDIO_MNIST_DIR, "AudioMNIST-master")
        extracted_data_dir = os.path.join(extracted_dir, "data")

        os.makedirs(AUDIO_MNIST_RAW_DIR, exist_ok=True)

        for item in os.listdir(extracted_data_dir):
            src_path = os.path.join(extracted_data_dir, item)
            dst_path = os.path.join(AUDIO_MNIST_RAW_DIR, item)

            if os.path.isdir(src_path) or item == AUDIO_MNIST_META_FILE:
                if os.path.exists(dst_path):
                    if os.path.isdir(dst_path):
                        shutil.rmtree(dst_path)
                    else:
                        os.remove(dst_path)
                shutil.move(src_path, dst_path)

        shutil.rmtree(extracted_dir)
        os.remove(zip_path)

        logger.info(f"Finished downloading AudioMNIST dataset to {AUDIO_MNIST_DIR}")
        return True

    def _load_data(self) -> None:
        self.audio_files = []
        self.labels = []

        # Iterate through speaker directories (01-60)
        for speaker_id in range(1, 61):
            speaker_dir = os.path.join(AUDIO_MNIST_RAW_DIR, f"{speaker_id:02d}")

            if not os.path.exists(speaker_dir):
                continue

            # Get all digit recordings for the current speaker
            for digit in range(10):
                for repetition in range(50):
                    filename = f"{digit}_{speaker_id:02d}_{repetition}.wav"
                    file_path = os.path.join(speaker_dir, filename)

                    if os.path.exists(file_path):
                        self.audio_files.append(file_path)
                        self.labels.append(digit)

        logger.info(f"Loaded {len(self.audio_files)} audio samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an audio sample and its label by index."""
        file_path = self.audio_files[idx]
        label = self.labels[idx]

        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        signal, sr = torchaudio.load(file_path, backend="soundfile")
        # signal = signal.to("mps")
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


class DataSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, label = self.dataset[self.indices[idx]]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


def create_audio_mnist_dataloaders(
    valid_size: int = 4500, test_size: int = 4500, download: bool = True, transform=None, num_workers: int = 4
):
    """
    Create data loaders for the AudioMNIST dataset.
      - Training set: 70% (21,000 recordings)
      - Validation set: 15% (4,500 recordings)
      - Test set: 15% (4,500 recordings)
    """
    dataset = AudioMNISTDataset(download=download)

    dataset_size = len(dataset)
    train_size = dataset_size - valid_size - test_size

    indices = torch.randperm(dataset_size, generator=torch.Generator()).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + valid_size]
    test_indices = indices[train_size + valid_size :]

    train_dataset = DataSubset(dataset, train_indices, transform)
    val_dataset = DataSubset(dataset, val_indices, transform)
    test_dataset = DataSubset(dataset, test_indices, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
