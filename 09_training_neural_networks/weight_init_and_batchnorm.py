"""
weight_init_and_batchnorm.py - CS231n Lecture 6 & 7: Weight Initialization & Batch Normalization
-------------------------------------------------------------------------------------------------
Visualizes & benchmarks:
1. Weight Initialization Dynamics (10-layer network):
   - Small std (0.01): Signal collapses to 0 across layers.
   - Large std (1.0): Signal saturates/explodes.
   - Xavier / Glorot: Optimal for Tanh, collapses for ReLU (variance halved per layer).
   - He / Kaiming: Preserves variance = 1 across all 10 layers with ReLU.
2. Batch Normalization (Ioffe & Szegedy, 2015):
   - Stabilizes activations regardless of initialization.
   - Enables training with 10x higher learning rate without diverging.
3. Learning Rate Schedules:
   - Cosine Annealing vs Step Decay vs Constant LR.
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


# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Chapter 09] Running on device: {device}")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Ensure figures output directory exists
os.makedirs("09_training_neural_networks/figures", exist_ok=True)


# ==============================================================================
# 1. Weight Initialization Dynamics Across 10 Layers
# ==============================================================================
def experiment_weight_initialization():
    print("\n--- 1. Simulating Weight Initialization Dynamics Across 10 Layers ---")
    num_layers = 10
    layer_dim = 512
    batch_size = 1000
    
    # Input data: standard normal
    X = np.random.randn(batch_size, layer_dim)
    
    schemes = [
        ("Small Gaussian (std=0.01, Tanh)", "tanh", 0.01, "small"),
        ("Large Gaussian (std=0.05, Tanh)", "tanh", 0.05, "large"),
        ("Xavier / Glorot (Tanh)", "tanh", None, "xavier"),
        ("Xavier on ReLU (Failure)", "relu", None, "xavier"),
        ("He / Kaiming (ReLU - Optimal)", "relu", None, "kaiming"),
    ]
    
    fig, axes = plt.subplots(len(schemes), num_layers, figsize=(18, 10), sharey=True)
    fig.suptitle("CS231n Lecture 6 : Distribution des Activations à travers 10 Couches", fontsize=16, fontweight="bold", y=0.98)
    
    for row_idx, (title, act_fn, scale, method) in enumerate(schemes):
        curr_x = X.copy()
        layer_means = []
        layer_stds = []
        
        for l in range(num_layers):
            Din, Dout = layer_dim, layer_dim
            if method == "small":
                W = np.random.randn(Din, Dout) * scale
            elif method == "large":
                W = np.random.randn(Din, Dout) * scale
            elif method == "xavier":
                # Xavier: std = sqrt(2 / (Din + Dout)) or sqrt(1 / Din)
                W = np.random.randn(Din, Dout) * np.sqrt(2.0 / (Din + Dout))
            elif method == "kaiming":
                # He/Kaiming: std = sqrt(2 / Din) specifically designed for ReLU
                W = np.random.randn(Din, Dout) * np.sqrt(2.0 / Din)
                
            curr_x = curr_x @ W
            
            if act_fn == "tanh":
                curr_x = np.tanh(curr_x)
            elif act_fn == "relu":
                curr_x = np.maximum(0, curr_x)
                
            layer_means.append(np.mean(curr_x))
            layer_stds.append(np.std(curr_x))
            
            ax = axes[row_idx, l]
            ax.hist(curr_x.ravel(), bins=30, range=(-1.5, 1.5) if act_fn == "tanh" else (-0.2, 2.0),
                    color="#0284c7" if "Optimal" in title or "Xavier / Glorot (Tanh)" in title else "#ef4444",
                    alpha=0.85, density=True)
            ax.set_yticks([])
            if row_idx == len(schemes) - 1:
                ax.set_xlabel(f"Couche {l+1}", fontsize=10)
            if l == 0:
                ax.set_ylabel(title, fontsize=9, fontweight="bold")
                
        print(f"[{title}] Couche 1: std={layer_stds[0]:.3f} -> Couche 10: std={layer_stds[-1]:.4f}")
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_path = "09_training_neural_networks/figures/weight_initialization_10layers.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Graphique sauvegardé : {save_path}")


# ==============================================================================
# 2. Batch Normalization Benchmark (Training Stability & Learning Rate Boost)
# ==============================================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 128, 128, 128], num_classes=10, use_bn=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def experiment_batchnorm_and_high_lr():
    print("\n--- 2. Batch Normalization Benchmark on Fashion-MNIST ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load dataset
    data_path = "./data"
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=False, transform=transform)
    
    # Fast subset for rapid iteration (3000 train, 1000 test)
    train_subset = Subset(train_dataset, range(3000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    configurations = [
        ("Sans BatchNorm (lr=0.01)", False, 0.01, "#64748b"),
        ("Avec BatchNorm (lr=0.01)", True, 0.01, "#0284c7"),
        ("Sans BatchNorm (lr=0.1 - Diverge)", False, 0.1, "#ef4444"),
        ("Avec BatchNorm (lr=0.1 - Rapide)", True, 0.1, "#10b981"),
    ]
    
    epochs = 8
    history = {}
    
    for name, use_bn, lr, color in configurations:
        print(f"Entraînement de : {name}...")
        model = DeepMLP(use_bn=use_bn).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        train_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                
            avg_loss = total_loss / len(train_subset)
            train_losses.append(min(avg_loss, 5.0)) # Clip for plotting divergence
            
            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    
            val_acc = (correct / len(test_subset)) * 100.0
            val_accs.append(val_acc)
            
        print(f"  -> Résultat final : Loss={train_losses[-1]:.3f}, Précision={val_accs[-1]:.2f}%")
        history[name] = {"loss": train_losses, "acc": val_accs, "color": color}
        
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for name, data in history.items():
        ls = "--" if "Diverge" in name else "-"
        plt.plot(range(1, epochs + 1), data["loss"], label=name, color=data["color"], linewidth=2.2, linestyle=ls, marker="o", markersize=4)
    plt.title("Perte d'Entraînement (Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Époque", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    
    plt.subplot(1, 2, 2)
    for name, data in history.items():
        ls = "--" if "Diverge" in name else "-"
        plt.plot(range(1, epochs + 1), data["acc"], label=name, color=data["color"], linewidth=2.2, linestyle=ls, marker="s", markersize=4)
    plt.title("Précision de Test (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Époque", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    
    plt.tight_layout()
    save_path = "09_training_neural_networks/figures/batchnorm_lr_benchmark.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Graphique sauvegardé : {save_path}")


# ==============================================================================
# 3. Learning Rate Schedules (Cosine Annealing vs StepLR vs Constant)
# ==============================================================================
def experiment_lr_schedules():
    print("\n--- 3. Visualizing Learning Rate Schedules ---")
    total_steps = 100
    base_lr = 0.1
    
    # 1. Constant
    lrs_constant = [base_lr] * total_steps
    
    # 2. Step Decay (drop by 0.5 every 25 steps)
    lrs_step = []
    curr = base_lr
    for s in range(total_steps):
        if s > 0 and s % 25 == 0:
            curr *= 0.5
        lrs_step.append(curr)
        
    # 3. Cosine Annealing: lr_t = 0.5 * lr_0 * (1 + cos(t * pi / T_max))
    lrs_cosine = [0.5 * base_lr * (1.0 + np.cos(s * np.pi / total_steps)) for s in range(total_steps)]
    
    # 4. Warmup + Cosine Annealing (Modern Standard)
    warmup_steps = 15
    lrs_warmup_cosine = []
    for s in range(total_steps):
        if s < warmup_steps:
            lrs_warmup_cosine.append(base_lr * (s + 1) / warmup_steps)
        else:
            t = s - warmup_steps
            T_max = total_steps - warmup_steps
            lrs_warmup_cosine.append(0.5 * base_lr * (1.0 + np.cos(t * np.pi / T_max)))
            
    plt.figure(figsize=(10, 5))
    plt.plot(lrs_constant, label=r"Constant LR ($\eta = 0.1$)", color="#64748b", linestyle="--", linewidth=1.8)
    plt.plot(lrs_step, label=r"Step Decay ($\gamma=0.5$ tous les 25 pas)", color="#f59e0b", linewidth=2.2)
    plt.plot(lrs_cosine, label="Cosine Annealing (Loshchilov & Hutter)", color="#0284c7", linewidth=2.2)
    plt.plot(lrs_warmup_cosine, label="Linear Warmup (15 pas) + Cosine Annealing", color="#10b981", linewidth=2.5)
    
    plt.title("CS231n Lecture 7 : Comparatif des Stratégies de Learning Rate (Schedules)", fontsize=13, fontweight="bold")
    plt.xlabel("Pas d'Optimisation (Iterations / Epochs)", fontsize=11)
    plt.ylabel(r"Learning Rate ($\eta$)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)
    
    plt.tight_layout()
    save_path = "09_training_neural_networks/figures/lr_schedules.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Graphique sauvegardé : {save_path}")


if __name__ == "__main__":
    t0 = time.time()
    experiment_weight_initialization()
    experiment_batchnorm_and_high_lr()
    experiment_lr_schedules()
    print(f"\n=== Chapitre 09 exécuté avec succès en {time.time() - t0:.2f}s ===")
