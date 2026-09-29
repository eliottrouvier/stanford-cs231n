"""
resnet_degradation_benchmark.py - CS231n Lecture 9: The ResNet Revolution & The Degradation Problem
----------------------------------------------------------------------------------------------------
Demonstrates Kaiming He et al.'s landmark 2015 discovery:
- Why deep plain ConvNets fail: not due to overfitting, but OPTIMIZATION DEGRADATION.
- Compares:
  1. Plain-20 ConvNet (20 layers, no shortcuts).
  2. ResNet-20 (20 layers, with residual identity shortcuts: y = F(x) + x).
- Evaluates training loss, validation accuracy, and gradient flow through time.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Chapter 10] Running on device: {device}")

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("10_cnn_architectures/figures", exist_ok=True)


# ==============================================================================
# Building Blocks: Plain Block vs Residual Block
# ==============================================================================
class PlainBlock(nn.Module):
    """2-conv layer plain block without residual shortcut."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out)


class ResidualBlock(nn.Module):
    """ResNet block with identity skip connection: y = F(x) + x."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Residual Addition: gradient highway
        out = out + identity
        return self.relu(out)


class DeepConvNet(nn.Module):
    """Deep 20-layer network (Input conv + 9 blocks of 2 convs + FC = 20 layers)."""
    def __init__(self, use_residual=False, num_classes=10, channels=32):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        block_cls = ResidualBlock if use_residual else PlainBlock
        # 9 blocks * 2 conv layers = 18 conv layers
        self.blocks = nn.Sequential(*[block_cls(channels) for _ in range(9)])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.blocks(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def run_degradation_benchmark():
    print("\n--- Benchmark : Plain-20 vs ResNet-20 sur Fashion-MNIST ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    data_path = "./data"
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=False, transform=transform)
    
    # 3,500 train, 1,000 test for fast convergence demonstration
    train_subset = Subset(train_dataset, range(3500))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    models = [
        ("Plain-20 (Sans Residual)", False, "#ef4444"),
        ("ResNet-20 (Avec Residual F(x)+x)", True, "#0284c7")
    ]
    
    epochs = 10
    history = {}
    
    for name, use_res, color in models:
        print(f"\nEntraînement de {name}...")
        model = DeepConvNet(use_residual=use_res, channels=32).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Nombre de paramètres : {total_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        train_accs = []
        val_accs = []
        
        for ep in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, lbls)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * imgs.size(0)
                correct_train += (preds.argmax(1) == lbls).sum().item()
                total_train += imgs.size(0)
                
            scheduler.step()
            
            ep_loss = running_loss / total_train
            ep_train_acc = (correct_train / total_train) * 100.0
            train_losses.append(ep_loss)
            train_accs.append(ep_train_acc)
            
            # Validation
            model.eval()
            correct_val = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs)
                    correct_val += (preds.argmax(1) == lbls).sum().item()
                    
            ep_val_acc = (correct_val / len(test_subset)) * 100.0
            val_accs.append(ep_val_acc)
            
            print(f"  Époque {ep+1:02d}/{epochs:02d} | Loss: {ep_loss:.3f} | Train Acc: {ep_train_acc:.1f}% | Val Acc: {ep_val_acc:.1f}%")
            
        history[name] = {
            "loss": train_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
            "color": color
        }
        
    # Plot degradation curves (Loss & Accuracy)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for name, d in history.items():
        ls = "--" if "Plain" in name else "-"
        plt.plot(range(1, epochs + 1), d["loss"], label=name, color=d["color"], linewidth=2.4, linestyle=ls, marker="o")
    plt.title("Perte d'Entraînement (Training Loss)", fontsize=12, fontweight="bold")
    plt.xlabel("Époque", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    
    plt.subplot(1, 3, 2)
    for name, d in history.items():
        ls = "--" if "Plain" in name else "-"
        plt.plot(range(1, epochs + 1), d["train_acc"], label=name, color=d["color"], linewidth=2.4, linestyle=ls, marker="^")
    plt.title("Précision d'Entraînement (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Époque", fontsize=10)
    plt.ylabel("Train Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    
    plt.subplot(1, 3, 3)
    for name, d in history.items():
        ls = "--" if "Plain" in name else "-"
        plt.plot(range(1, epochs + 1), d["val_acc"], label=name, color=d["color"], linewidth=2.4, linestyle=ls, marker="s")
    plt.title("Précision de Test (Validation Accuracy %)", fontsize=12, fontweight="bold")
    plt.xlabel("Époque", fontsize=10)
    plt.ylabel("Test Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    
    plt.tight_layout()
    save_path = "10_cnn_architectures/figures/resnet_vs_plain_degradation.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n-> Graphique de dégradation sauvegardé : {save_path}")


if __name__ == "__main__":
    t0 = time.time()
    run_degradation_benchmark()
    print(f"=== Fin du benchmark ResNet en {time.time() - t0:.2f}s ===")
