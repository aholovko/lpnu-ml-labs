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
from pathlib import Path
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
AUDIO_MNIST_DIR = DATA_DIR / "AUDIOMNIST"
AUDIO_MNIST_RAW_DIR = AUDIO_MNIST_DIR / "raw"


class AudioMNISTDataset(Dataset):
    """AudioMNIST dataset."""

    def __init__(self, download: bool = True, num_samples: int = SAMPLE_RATE):
        self.num_samples = num_samples

        if download:
            self._download_dataset_if_needed()

        self._load_data()

    @staticmethod
    def _download_dataset_if_needed() -> bool:
        meta_file_path = AUDIO_MNIST_DIR / AUDIO_MNIST_META_FILE

        if meta_file_path.exists():
            return True

        os.makedirs(AUDIO_MNIST_DIR, exist_ok=True)

        zip_path = AUDIO_MNIST_DIR / "audiomnist.zip"
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

        extracted_dir = AUDIO_MNIST_DIR / "AudioMNIST-master"
        extracted_data_dir = extracted_dir / "data"

        os.makedirs(AUDIO_MNIST_RAW_DIR, exist_ok=True)

        for item in os.listdir(extracted_data_dir):
            src_path = extracted_data_dir / item
            dst_path = AUDIO_MNIST_RAW_DIR / item

            if src_path.is_dir() or item == AUDIO_MNIST_META_FILE:
                if dst_path.exists():
                    if dst_path.is_dir():
                        shutil.rmtree(dst_path)
                    else:
                        os.remove(dst_path)
                shutil.move(str(src_path), str(dst_path))

        shutil.rmtree(extracted_dir)
        os.remove(zip_path)

        logger.info(f"Finished downloading AudioMNIST dataset to {AUDIO_MNIST_DIR}")
        return True

    def _load_data(self) -> None:
        self.audio_files = []
        self.labels = []

        # Iterate through speaker directories (01-60)
        for speaker_id in range(1, 61):
            speaker_dir = AUDIO_MNIST_RAW_DIR / f"{speaker_id:02d}"

            if not speaker_dir.exists():
                continue

            # Get all digit recordings for the current speaker
            for digit in range(10):
                for repetition in range(50):
                    filename = f"{digit}_{speaker_id:02d}_{repetition}.wav"
                    file_path = speaker_dir / filename

                    if file_path.exists():
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

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

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


class AudioMNISTDataModule:
    """Data module for the AudioMNIST dataset."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        valid_size: int = 4500,
        test_size: int = 4500,
        transform=None,
        num_workers: int = 4,
        download: bool = True,
    ):
        """Initialize the AudioMNIST data module.

        Args:
            data_dir: Directory where the dataset will be stored
            batch_size: Batch size for dataloaders
            valid_size: Number of samples to use for validation
            test_size: Number of samples to use for test
            transform: Optional transform to apply to the data
            num_workers: Number of workers for data loading
            download: Whether to download the dataset if not found
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.transform = transform
        self.num_workers = num_workers
        self.download = download
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.dataset = None

    def prepare_data(self) -> None:
        """Download or prepare the AudioMNIST dataset."""
        if self.download:
            # Create the dataset which will trigger download if needed
            AudioMNISTDataset(download=True)

    def setup(self) -> None:
        """Setup train, validation, and test datasets."""
        self.dataset = AudioMNISTDataset(download=False)

        # Create splits
        dataset_size = len(self.dataset)
        train_size = dataset_size - self.valid_size - self.test_size

        indices = torch.randperm(dataset_size, generator=torch.Generator()).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + self.valid_size]
        test_indices = indices[train_size + self.valid_size :]

        self.train_dataset = DataSubset(self.dataset, train_indices, self.transform)
        self.valid_dataset = DataSubset(self.dataset, val_indices, self.transform)
        self.test_dataset = DataSubset(self.dataset, test_indices, self.transform)

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        if self.valid_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Call setup() before accessing dataloaders")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
