"""
AudioMNIST Dataset

A spoken digits dataset consisting of 30,000 audio samples of digits (0-9) recorded by
60 different speakers (30 female, 30 male). Each speaker contributed 50 recordings of
each digit, resulting in a balanced dataset (500 recordings per speaker).

Dataset Structure:
- 10 digits (0-9)
- 60 speakers (labeled 01-60)
- 50 recordings per digit per speaker (labeled 0-49)
- Audio format: WAV files (8kHz, mono, 16-bit)

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
from urllib import request

from torch.utils.data import Dataset

from src.paths import DATA_DIR
from src.utils import setup_logging

logger = setup_logging(logging.INFO)

AUDIO_MNIST_BASE_URL = "https://raw.githubusercontent.com/soerenab/AudioMNIST/refs/heads/master/data/"
AUDIO_MNIST_META_FILE = "audioMNIST_meta.txt"
AUDIO_MNIST_DIR = os.path.join(DATA_DIR, "AUDIOMNIST", "raw")


class AudioMNISTDataset(Dataset):
    """AudioMNIST dataset."""

    def __init__(self, download: bool = True):
        if download:
            self._download_dataset_if_needed()

        self._load_data()

    @staticmethod
    def _download_dataset_if_needed() -> bool:
        meta_file_path = os.path.join(AUDIO_MNIST_DIR, AUDIO_MNIST_META_FILE)

        if os.path.exists(meta_file_path):
            return True

        os.makedirs(AUDIO_MNIST_DIR, exist_ok=True)

        try:
            request.urlretrieve(AUDIO_MNIST_BASE_URL + AUDIO_MNIST_META_FILE, meta_file_path)
            logger.info(f"Downloaded metadata to {meta_file_path}")
        except Exception as e:
            logger.error(f"Failed to download metadata: {e}")
            return False

        for speaker_id in range(1, 61):
            speaker_id_padded = f"{speaker_id:02d}"
            speaker_dir = os.path.join(AUDIO_MNIST_DIR, speaker_id_padded)
            os.makedirs(speaker_dir, exist_ok=True)

            for digit in range(10):
                for repetition in range(50):
                    filename = f"{digit}_{speaker_id_padded}_{repetition}.wav"
                    file_url = f"{AUDIO_MNIST_BASE_URL}{speaker_id_padded}/{filename}"
                    file_path = os.path.join(speaker_dir, filename)

                    if os.path.exists(file_path):
                        continue

                    try:
                        request.urlretrieve(file_url, file_path)
                        logger.info(f"Downloaded {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to download {file_url}: {e}")
                        continue

        logger.info(f"Finished downloading AudioMNIST dataset to {AUDIO_MNIST_DIR}")
        return True

    def _load_data(self) -> None:
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass
