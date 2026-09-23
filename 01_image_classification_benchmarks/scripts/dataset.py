"""Module de chargement et de préparation du jeu de données Fashion-MNIST."""

from pathlib import Path
from typing import Optional
import numpy as np
import torchvision

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist(
    root: str | Path = "data",
    num_train: int = 10000,
    num_test: int = 2000,
    seed: int = 42,
) -> dict[str, np.ndarray | list[str]]:
    """Télécharge et prépare Fashion-MNIST avec normalisation [0, 1] et mise en forme vectorisée.

    Args:
        root: Répertoire racine des données (défaut: 'data').
        num_train: Nombre d'échantillons d'entraînement (max 60000).
        num_test: Nombre d'échantillons de test (max 10000).
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Dictionnaire contenant:
            - 'X_train_flat': (N_train, 784) en float32 normalisé [0, 1]
            - 'X_train_img': (N_train, 28, 28) en uint8 pour la visualisation
            - 'y_train': (N_train,) labels entiers [0-9]
            - 'X_test_flat': (N_test, 784)
            - 'X_test_img': (N_test, 28, 28)
            - 'y_test': (N_test,)
            - 'classes': Liste des 10 classes
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    train_set = torchvision.datasets.FashionMNIST(root=str(root_path), train=True, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=str(root_path), train=False, download=True)

    # Récupération sous forme numpy
    X_train_raw = train_set.data.numpy()
    y_train_raw = np.array(train_set.targets.numpy(), dtype=np.int64)

    X_test_raw = test_set.data.numpy()
    y_test_raw = np.array(test_set.targets.numpy(), dtype=np.int64)

    # Sous-échantillonnage stratifié ou avec seed
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(X_train_raw), size=min(num_train, len(X_train_raw)), replace=False)
    test_idx = rng.choice(len(X_test_raw), size=min(num_test, len(X_test_raw)), replace=False)

    X_train_img = X_train_raw[train_idx]
    y_train = y_train_raw[train_idx]

    X_test_img = X_test_raw[test_idx]
    y_test = y_test_raw[test_idx]

    # Normalisation [0, 1] et aplatissement
    X_train_flat = X_train_img.reshape(X_train_img.shape[0], -1).astype(np.float32) / 255.0
    X_test_flat = X_test_img.reshape(X_test_img.shape[0], -1).astype(np.float32) / 255.0

    return {
        "X_train_flat": X_train_flat,
        "X_train_img": X_train_img,
        "y_train": y_train,
        "X_test_flat": X_test_flat,
        "X_test_img": X_test_img,
        "y_test": y_test,
        "classes": FASHION_CLASSES,
    }
