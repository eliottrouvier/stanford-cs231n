"""Utilitaires de téléchargement et de chargement du jeu de données CIFAR-10."""

from pathlib import Path
from typing import Optional
import numpy as np
import torchvision
import torchvision.transforms as transforms

CIFAR10_CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10(
    root: str | Path = "data",
    num_train: int = 49000,
    num_val: int = 1000,
    num_test: int = 10000,
    flatten: bool = True,
    subtract_mean: bool = True,
) -> dict[str, np.ndarray]:
    """Télécharge (si nécessaire) et charge le dataset CIFAR-10 avec division train/val/test.

    Args:
        root: Dossier où stocker les données brutes (ignoré par Git).
        num_train: Nombre d'images d'entraînement (max 50000 - num_val).
        num_val: Nombre d'images de validation.
        num_test: Nombre d'images de test (max 10000).
        flatten: Si True, aplatit les images en vecteurs 1D (N, 3072).
                 Si False, conserve la forme (N, 32, 32, 3).
        subtract_mean: Si True, soustrait l'image moyenne du jeu d'entraînement.

    Returns:
        Dictionnaire contenant 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'mean_image'.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Téléchargement via torchvision
    cifar10_train = torchvision.datasets.CIFAR10(root=str(root_path), train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=str(root_path), train=False, download=True)

    X_train_raw = np.array(cifar10_train.data, dtype=np.float32)
    y_train_raw = np.array(cifar10_train.targets, dtype=np.int64)

    X_test_raw = np.array(cifar10_test.data, dtype=np.float32)
    y_test_raw = np.array(cifar10_test.targets, dtype=np.int64)

    # Découpage train / validation
    X_train = X_train_raw[:num_train]
    y_train = y_train_raw[:num_train]
    X_val = X_train_raw[num_train : num_train + num_val]
    y_val = y_train_raw[num_train : num_train + num_val]

    X_test = X_test_raw[:num_test]
    y_test = y_test_raw[:num_test]

    mean_image = None
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "mean_image": mean_image,
        "classes": CIFAR10_CLASSES,
    }
