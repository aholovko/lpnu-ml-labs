"""
Feature engineering for the MNIST dataset.
"""

import torchvision.transforms as transforms


def normalize_mnist(x, mean=0.1307, std=0.3081):
    """
    Normalize MNIST images using the dataset's mean and standard deviation.
    """

    return transforms.Normalize(mean=[mean], std=[std])(x)


def extract_features(x):
    """
    Generic feature extraction function that can apply various transformations.
    """

    x = normalize_mnist(x)
    return x
