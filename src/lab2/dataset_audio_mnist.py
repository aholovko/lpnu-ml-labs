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
import shutil
import zipfile
from urllib import request

from torch.utils.data import Dataset
from tqdm import tqdm

from src.paths import DATA_DIR
from src.utils import setup_logging

logger = setup_logging(logging.INFO)


AUDIO_MNIST_ZIP_URL = "https://github.com/soerenab/AudioMNIST/archive/refs/heads/master.zip"
AUDIO_MNIST_META_FILE = "audioMNIST_meta.txt"
AUDIO_MNIST_DIR = os.path.join(DATA_DIR, "AUDIOMNIST")
AUDIO_MNIST_RAW_DIR = os.path.join(AUDIO_MNIST_DIR, "raw")


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
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass
