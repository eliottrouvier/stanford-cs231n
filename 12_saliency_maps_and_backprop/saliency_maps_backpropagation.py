"""
Stanford CS231n - Lecture 12 : Visualizing and Understanding Convolutional Networks
Saliency Maps via Backpropagation (Simonyan, Vedaldi, Zisserman, 2013)
& SmoothGrad (Smilkov et al., 2017)

Ce script implémente de zéro le calcul de cartes de saillance (saliency maps) par rétropropagation,
le débruitage par SmoothGrad, la localisation d'objets faiblement supervisée (Weakly Supervised Localization)
et la saillance contrastive multi-classes sur le modèle canonique CS231n (SqueezeNet 1.1).
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

# Définition des répertoires
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "sample_images"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Normalisation standard ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

deprocess_transform = T.Compose([
    T.Normalize(mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
                std=[1.0 / s for s in IMAGENET_STD]),
    T.Lambda(lambda t: torch.clamp(t, 0.0, 1.0))
])


def load_model(device: torch.device):
    """Charge le modèle SqueezeNet 1.1 pré-entraîné (modèle officiel du TP CS231n)."""
    weights = models.SqueezeNet1_1_Weights.DEFAULT
    model = models.squeezenet1_1(weights=weights).to(device)
    model.eval()
    categories = weights.meta["categories"]
    return model, categories


def preprocess_image(image_path: Path, device: torch.device):
    """Charge et prétraite une image en tenseur normalisé (1, 3, 224, 224)."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess_transform(img).unsqueeze(0).to(device)
    return img, tensor


def compute_saliency_map(model: torch.nn.Module, X: torch.Tensor, target_class: int = None):
    """
    Calcule la carte de saillance vanilla d'une image X par rétropropagation.

    Mathématiques :
        Soit S_c(I) le logit non normalisé de la classe cible c.
        w = ∂S_c / ∂I ∈ R^(3 x H x W)
        M(x, y) = max_{c ∈ {R, G, B}} |w(c, x, y)|

    Args:
        model: Réseau de neurones en mode eval()
        X: Tenseur de l'image (1, 3, H, W)
        target_class: Indice de la classe cible (si None, utilise argmax des logits)

    Returns:
        saliency_map: NumPy array (H, W) normalisé dans [0, 1]
        target_class: Indice de classe utilisé
        score: Score non normalisé du logit
    """
    # Clone avec graphe de calcul pour préserver l'entrée
    X_input = X.clone().detach().requires_grad_(True)
    scores = model(X_input)

    if target_class is None:
        target_class = scores.argmax(dim=1).item()

    score = scores[0, target_class]
    score.backward()

    # w ∈ (1, 3, H, W)
    grad = X_input.grad.detach()

    # M(x, y) = max_c |w_c(x, y)|
    saliency, _ = torch.max(grad.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()

    # Normalisation Min-Max [0, 1]
    denom = saliency.max() - saliency.min()
    if denom > 1e-8:
        saliency = (saliency - saliency.min()) / denom

    return saliency, target_class, score.item()


def compute_smoothgrad(model: torch.nn.Module, X: torch.Tensor, target_class: int = None,
                       stdev_spread: float = 0.15, n_samples: int = 30):
    """
    Calcule la carte de saillance débruitée par SmoothGrad (Smilkov et al., 2017).

    Mathématiques :
        M_smooth(I) = 1/N ∑_{i=1}^N M(I + ε_i),  ε_i ~ N(0, σ^2)
        avec σ = stdev_spread * (I_max - I_min)

    L'espérance sous bruit gaussien lisse les fluctuations locales des ReLUs par morceaux.
    """
    x_min, x_max = X.min().item(), X.max().item()
    sigma = stdev_spread * (x_max - x_min)

    smooth_saliency = np.zeros((X.shape[2], X.shape[3]), dtype=np.float32)

    for _ in range(n_samples):
        noise = torch.randn_like(X) * sigma
        noisy_x = (X + noise).clone().detach().requires_grad_(True)
        scores = model(noisy_x)

        if target_class is None:
            target_class = scores.argmax(dim=1).item()

        score = scores[0, target_class]
        score.backward()

        grad = noisy_x.grad.detach()
        sample_saliency, _ = torch.max(grad.abs(), dim=1)
        smooth_saliency += sample_saliency[0].cpu().numpy()

    smooth_saliency /= n_samples

    denom = smooth_saliency.max() - smooth_saliency.min()
    if denom > 1e-8:
        smooth_saliency = (smooth_saliency - smooth_saliency.min()) / denom

    return smooth_saliency, target_class


def weakly_supervised_bbox(saliency_map: np.ndarray, percentile: float = 95.0):
    """
    Extrait un masque binaire et une boîte englobante (Bounding Box)
    sans aucune supervision d'annotation de boîte pendant l'entraînement.
    """
    thresh = np.percentile(saliency_map, percentile)
    mask = saliency_map >= thresh

    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return mask, (0, 0, saliency_map.shape[1], saliency_map.shape[0])

    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    return mask, (xmin, ymin, xmax - xmin, ymax - ymin)


# =====================================================================
# Génération des Figures Expérimentales
# =====================================================================

def plot_saliency_gallery(model, categories, device):
    """Figure 1 : Galerie comparative Image / Saliency Vanilla / SmoothGrad / Overlay."""
    print("▶ Génération Figure 1 : Galerie de cartes de saillance...")
    image_files = ["dog.jpg", "tiger.jpg", "sports_car.jpg", "spider.png"]
    valid_images = [f for f in image_files if (IMAGES_DIR / f).exists()]

    fig, axes = plt.subplots(len(valid_images), 4, figsize=(16, 4 * len(valid_images)))
    if len(valid_images) == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, fname in enumerate(valid_images):
        img_path = IMAGES_DIR / fname
        _, X = preprocess_image(img_path, device)

        # Calcul Saliency Vanilla & SmoothGrad
        vanilla_sal, pred_id, _ = compute_saliency_map(model, X)
        smooth_sal, _ = compute_smoothgrad(model, X, target_class=pred_id, n_samples=25)

        class_name = categories[pred_id]
        img_rgb = Image.open(img_path).convert("RGB").resize((224, 224))

        # 1. Image originale
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"Image ({fname})\nPrédiction: {class_name}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Saliency Vanilla
        im1 = axes[row_idx, 1].imshow(vanilla_sal, cmap="hot")
        axes[row_idx, 1].set_title(f"Saliency Map (Vanilla Backprop)\nmax_c |∂S_{{{pred_id}}} / ∂I_c|", fontsize=11)
        axes[row_idx, 1].axis("off")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 3. SmoothGrad
        im2 = axes[row_idx, 2].imshow(smooth_sal, cmap="hot")
        axes[row_idx, 2].set_title("SmoothGrad (N=25, σ=0.15)\nDébruitage Gaussien", fontsize=11)
        axes[row_idx, 2].axis("off")
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # 4. Superposition (Overlay)
        axes[row_idx, 3].imshow(img_rgb)
        axes[row_idx, 3].imshow(smooth_sal, cmap="jet", alpha=0.45)
        axes[row_idx, 3].set_title("Incrustation Saillance / Image\nZones discriminantes", fontsize=11)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "saliency_maps_gallery.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")


def plot_smoothgrad_denoising(model, categories, device):
    """Figure 2 : Impact du nombre d'échantillons N dans SmoothGrad."""
    print("▶ Génération Figure 2 : Débruitage progressif SmoothGrad...")
    sample_file = IMAGES_DIR / "dog.jpg"
    if not sample_file.exists():
        sample_file = list(IMAGES_DIR.glob("*.jpg"))[0]

    _, X = preprocess_image(sample_file, device)
    img_rgb = Image.open(sample_file).convert("RGB").resize((224, 224))

    sample_counts = [1, 5, 15, 30, 60]
    fig, axes = plt.subplots(1, len(sample_counts) + 1, figsize=(20, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Image d'Entrée (Dog)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, N in enumerate(sample_counts):
        if N == 1:
            sal, pred_id, _ = compute_saliency_map(model, X)
            title = f"N = 1 (Vanilla)\nBruit haute-fréquence"
        else:
            sal, _ = compute_smoothgrad(model, X, target_class=pred_id, n_samples=N)
            title = f"SmoothGrad (N = {N})\nσ = 0.15"

        im = axes[idx + 1].imshow(sal, cmap="inferno")
        axes[idx + 1].set_title(title, fontsize=11)
        axes[idx + 1].axis("off")

    plt.suptitle("Élimination du bruit de gradient par moyennage stochastique (SmoothGrad)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "smoothgrad_denoising_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")


def plot_class_contrastive(model, categories, device):
    """Figure 3 : Saillance contrastive sur une image composite (Cat + Dog)."""
    print("▶ Génération Figure 3 : Saillance contrastive multi-classes...")
    cat_dog_path = IMAGES_DIR / "cat_dog.jpg"
    if not cat_dog_path.exists():
        print("  ✗ Image cat_dog.jpg absente, étape ignorée.")
        return

    _, X = preprocess_image(cat_dog_path, device)
    img_rgb = Image.open(cat_dog_path).convert("RGB").resize((224, 224))

    # Classes cibles : 282 = tiger cat, 243 = bull mastiff
    cat_class_id = 282
    dog_class_id = 243

    sal_cat, _ = compute_smoothgrad(model, X, target_class=cat_class_id, n_samples=30)
    sal_dog, _ = compute_smoothgrad(model, X, target_class=dog_class_id, n_samples=30)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # 1. Image originale
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image Composite\nChat (gauche) + Chien (droite)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Cible = Chat
    axes[1].imshow(img_rgb)
    axes[1].imshow(sal_cat, cmap="jet", alpha=0.55)
    axes[1].set_title(f"Target = '{categories[cat_class_id]}'\nFocus net sur le chat", fontsize=12, color="darkblue", fontweight="bold")
    axes[1].axis("off")

    # 3. Cible = Chien
    axes[2].imshow(img_rgb)
    axes[2].imshow(sal_dog, cmap="jet", alpha=0.55)
    axes[2].set_title(f"Target = '{categories[dog_class_id]}'\nFocus net sur le chien", fontsize=12, color="darkred", fontweight="bold")
    axes[2].axis("off")

    # 4. Carte de différence contrastive
    diff_map = sal_cat - sal_dog
    im = axes[3].imshow(diff_map, cmap="bwr", vmin=-1, vmax=1)
    axes[3].set_title("Carte Différentielle\nBleu = Chat | Rouge = Chien", fontsize=12, fontweight="bold")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle("Preuve de Sélectivité de Classe : Même image, deux rétropropagations distinctes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "class_contrastive_cat_dog.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")


def plot_weakly_supervised_localization(model, categories, device):
    """Figure 4 : Localisation d'objet sans label de boîte (Weakly Supervised BBox)."""
    print("▶ Génération Figure 4 : Localisation faiblement supervisée...")
    sample_files = ["tiger.jpg", "dog.jpg"]
    valid = [f for f in sample_files if (IMAGES_DIR / f).exists()]

    fig, axes = plt.subplots(len(valid), 4, figsize=(18, 4.5 * len(valid)))
    if len(valid) == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, fname in enumerate(valid):
        img_path = IMAGES_DIR / fname
        _, X = preprocess_image(img_path, device)
        img_rgb = Image.open(img_path).convert("RGB").resize((224, 224))

        sal, pred_id = compute_smoothgrad(model, X, n_samples=30)
        mask, bbox = weakly_supervised_bbox(sal, percentile=93.0)

        # 1. Image
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"Entrée : {categories[pred_id]}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Carte de saillance
        im1 = axes[row_idx, 1].imshow(sal, cmap="hot")
        axes[row_idx, 1].set_title("Carte de Saillance Continue", fontsize=11)
        axes[row_idx, 1].axis("off")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 3. Masque binaire
        axes[row_idx, 2].imshow(mask, cmap="gray")
        axes[row_idx, 2].set_title("Masque Binaire (Seuil 93%)", fontsize=11)
        axes[row_idx, 2].axis("off")

        # 4. Bounding Box extraite
        axes[row_idx, 3].imshow(img_rgb)
        xmin, ymin, w, h = bbox
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2.5, edgecolor="cyan", facecolor="none")
        axes[row_idx, 3].add_patch(rect)
        axes[row_idx, 3].set_title(f"Boîte Détectée (Zero Supervision)\nBBox: [{xmin}, {ymin}, {xmin+w}, {ymin+h}]", fontsize=11, color="cyan", fontweight="bold")
        axes[row_idx, 3].axis("off")

    plt.suptitle("Localisation d'Objets Faiblement Supervisée (Simonyan et al., 2013)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "weakly_supervised_bbox_localization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")


def main():
    print("=" * 70)
    print("  CS231n - Module 12 : Saliency Maps via Backpropagation")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    model, categories = load_model(device)
    print(f"Modèle SqueezeNet 1.1 chargé avec {len(categories)} catégories ImageNet.")

    plot_saliency_gallery(model, categories, device)
    plot_smoothgrad_denoising(model, categories, device)
    plot_class_contrastive(model, categories, device)
    plot_weakly_supervised_localization(model, categories, device)

    print("\n✅ Toutes les visualisations de cartes de saillance ont été générées avec succès !")


if __name__ == "__main__":
    main()
