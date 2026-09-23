"""Utilitaires de visualisation graphique pour le projet CS231n.

Fournit des fonctions réutilisables pour configurer les styles,
tracer les frontières de décision 2D, afficher des grilles d'images et
sauvegarder les figures proprement.
"""

from pathlib import Path
from typing import Callable, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    """Configure un style de tracé propre et esthétique."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "lines.linewidth": 2,
        "figure.autolayout": True,
    })


def save_figure(fig: plt.Figure, filepath: Path | str, dpi: int = 300) -> Path:
    """Sauvegarde une figure matplotlib dans un chemin spécifié en créant les répertoires si nécessaire."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"📊 Figure sauvegardée : {out_path}")
    return out_path


def plot_decision_boundary_2d(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Frontière de décision",
    filepath: Optional[Path | str] = None,
    resolution: int = 250,
) -> plt.Figure:
    """Trace et visualise la frontière de décision d'un classifieur sur un espace 2D.

    Args:
        predict_fn: Fonction prenant un tableau (N, 2) et renvoyant les prédictions (N,).
        X: Points de données 2D de forme (N, 2).
        y: Labels des classes de forme (N,).
        title: Titre de la figure.
        filepath: Chemin optionnel pour sauvegarder le fichier PNG.
        resolution: Résolution du maillage de grille.
    """
    set_plot_style()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = predict_fn(grid_points)
    preds = preds.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, preds, alpha=0.35, cmap="Set1")
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="Set1",
        edgecolor="k",
        s=35,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("Caractéristique $x_1$")
    ax.set_ylabel("Caractéristique $x_2$")
    plt.colorbar(contour, ax=ax, label="Classe prédite")

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_weight_templates(
    weights: np.ndarray,
    class_names: Sequence[str],
    img_shape: tuple[int, int, int] = (32, 32, 3),
    title: str = "Templates visuels des poids W",
    filepath: Optional[Path | str] = None,
) -> plt.Figure:
    """Visualise les lignes/colonnes de poids W d'un classifieur linéaire sous forme d'images templates.

    Args:
        weights: Tableau de forme (num_classes, D) où D = prod(img_shape).
        class_names: Noms des classes.
        img_shape: (Hauteur, Largeur, Canaux).
        title: Titre du tracé.
        filepath: Chemin optionnel pour sauvegarder la figure.
    """
    set_plot_style()
    num_classes = len(class_names)
    cols = min(5, num_classes)
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.8 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_classes):
        w = weights[i].reshape(img_shape)
        # Normalisation min-max pour étirer les valeurs entre [0, 1]
        w_min, w_max = w.min(), w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)

        axes[i].imshow(w_norm)
        axes[i].set_title(class_names[i])
        axes[i].axis("off")

    for j in range(num_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    if filepath:
        save_figure(fig, filepath)

    return fig
