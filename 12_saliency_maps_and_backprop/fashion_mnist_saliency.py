"""
Stanford CS231n - Module 12 : Saliency Maps sur Fashion-MNIST
Visualisation de la saillance par rétropropagation sur le jeu de données central du dépôt.

Ce script entraîne rapidement un ConvNet régularisé sur Fashion-MNIST,
puis calcule et visualise les zones discriminantes pour 8 classes de vêtements
(ex: semelle d'une basket vs tige d'une bottine vs anse d'un sac).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR.parent / "data"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


class FashionConvNet(nn.Module):
    """ConvNet compact à 3 couches convolutives avec Batch Normalization."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


def train_model(model, train_loader, device, epochs=3):
    """Entraîne rapidement le modèle pour obtenir des gradients sémantiques riches."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f"Entraînement rapide du ConvNet ({epochs} époques)...")
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        acc = 100.0 * correct / total
        print(f"  Époque [{epoch+1}/{epochs}] - Loss: {total_loss/total:.4f} | Précision: {acc:.2f}%")

    model.eval()
    return model


def compute_fashion_saliency(model, X, target_class):
    """Calcule la carte de saillance par rétropropagation pour un tenseur 1D canal."""
    X_input = X.clone().detach().requires_grad_(True)
    scores = model(X_input)
    score = scores[0, target_class]
    score.backward()

    # w ∈ (1, 1, 28, 28)
    grad = X_input.grad.detach()
    saliency = grad.abs()[0, 0].cpu().numpy()

    # Normalisation
    denom = saliency.max() - saliency.min()
    if denom > 1e-8:
        saliency = (saliency - saliency.min()) / denom
    return saliency


def main():
    print("=" * 70)
    print("  CS231n - Cartes de Saillance sur Fashion-MNIST")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Entraînement ou instanciation
    model = FashionConvNet().to(device)
    model = train_model(model, train_loader, device, epochs=3)

    # Sélection de classes emblématiques
    target_classes = [0, 1, 7, 8, 9]  # T-shirt, Trouser, Sneaker, Bag, Ankle boot
    samples = {}
    for X, y in test_data:
        if y in target_classes and y not in samples:
            samples[y] = X
        if len(samples) == len(target_classes):
            break

    print("\n▶ Génération de la figure de saillance Fashion-MNIST...")
    fig, axes = plt.subplots(len(target_classes), 4, figsize=(15, 3 * len(target_classes)))

    for idx, class_id in enumerate(target_classes):
        X = samples[class_id].unsqueeze(0).to(device)
        class_name = CLASS_NAMES[class_id]

        sal = compute_fashion_saliency(model, X, class_id)
        raw_img = samples[class_id].squeeze().numpy()

        # 1. Image originale
        axes[idx, 0].imshow(raw_img, cmap="gray")
        axes[idx, 0].set_title(f"Original : {class_name}", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        # 2. Carte de saillance (hot)
        im1 = axes[idx, 1].imshow(sal, cmap="hot")
        axes[idx, 1].set_title(f"Saillance |∂S/∂I|\nFocus du réseau", fontsize=11)
        axes[idx, 1].axis("off")
        fig.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # 3. Masque de décision (Top 10% pixels)
        thresh = np.percentile(sal, 90)
        mask = (sal >= thresh).astype(float)
        axes[idx, 2].imshow(mask, cmap="inferno")
        axes[idx, 2].set_title("Top 10% Pixels Clés\n(Signature sémantique)", fontsize=11)
        axes[idx, 2].axis("off")

        # 4. Superposition
        axes[idx, 3].imshow(raw_img, cmap="gray")
        axes[idx, 3].imshow(sal, cmap="jet", alpha=0.55)
        axes[idx, 3].set_title("Incrustation Image + Saillance", fontsize=11)
        axes[idx, 3].axis("off")

    plt.suptitle("Cartes de Saillance ConvNet sur Fashion-MNIST : Anatomie des Décisions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / "fashion_mnist_saliency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")
    print("\n✅ Analyse Fashion-MNIST terminée avec succès !")


if __name__ == "__main__":
    main()
