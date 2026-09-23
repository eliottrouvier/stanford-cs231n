"""Générateurs de données 2D synthétiques pour tester les classifieurs visuellement.

Inclut notamment le jeu de données "spirale" emblématique de CS231n (Andrej Karpathy)
ainsi que des blobs gaussiens et cercles concentriques.
"""

import numpy as np


def make_spiral_data(
    points_per_class: int = 100,
    num_classes: int = 3,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère le dataset en spirale classique de CS231n.

    Args:
        points_per_class: Nombre de points par branche/classe.
        num_classes: Nombre de branches spirales.
        noise: Amplitude du bruit gaussien.
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        X: Tableau 2D de forme (points_per_class * num_classes, 2).
        y: Labels des classes (points_per_class * num_classes,).
    """
    rng = np.random.default_rng(seed)
    total_points = points_per_class * num_classes
    X = np.zeros((total_points, 2), dtype=np.float32)
    y = np.zeros(total_points, dtype=np.int64)

    for c in range(num_classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)  # rayon
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, noise, points_per_class)  # thêta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c

    return X, y


def make_gaussian_clusters(
    points_per_cluster: int = 100,
    num_classes: int = 3,
    cluster_std: float = 0.45,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère des clusters gaussiens 2D bien ou modérément séparables."""
    rng = np.random.default_rng(seed)
    total_points = points_per_cluster * num_classes
    X = np.zeros((total_points, 2), dtype=np.float32)
    y = np.zeros(total_points, dtype=np.int64)

    # Répartir les centres en cercle
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    centers = np.c_[np.cos(angles) * 1.5, np.sin(angles) * 1.5]

    for c in range(num_classes):
        ix = range(points_per_cluster * c, points_per_cluster * (c + 1))
        X[ix] = rng.normal(loc=centers[c], scale=cluster_std, size=(points_per_cluster, 2))
        y[ix] = c

    return X, y


def make_concentric_circles(
    points_per_class: int = 150,
    num_classes: int = 2,
    noise: float = 0.08,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère des cercles concentriques (idéal pour montrer la non-linéarité)."""
    rng = np.random.default_rng(seed)
    total_points = points_per_class * num_classes
    X = np.zeros((total_points, 2), dtype=np.float32)
    y = np.zeros(total_points, dtype=np.int64)

    radii = np.linspace(0.4, 1.2, num_classes)
    for c, r in enumerate(radii):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        theta = rng.uniform(0, 2 * np.pi, points_per_class)
        radial_noise = rng.normal(0, noise, points_per_class)
        X[ix] = np.c_[(r + radial_noise) * np.cos(theta), (r + radial_noise) * np.sin(theta)]
        y[ix] = c

    return X, y
