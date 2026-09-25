"""
CS231n Lecture 5: ConvNet vs. Fully Connected MLP Benchmark on Fashion-MNIST
----------------------------------------------------------------------------
Trains on a fast, lightweight local subset (6,000 train samples, 1,000 test samples)
with ZERO network downloads:
1. Fully-Connected MLP (784 -> 128 -> 64 -> 10, 109,386 params)
2. Modern ConvNet (Conv2D -> BN -> ReLU -> MaxPool -> Conv2D -> BN -> ReLU -> MaxPool -> FC, 55,466 params)

Generates:
- Training loss and validation accuracy curves (showing CNN's sample-efficiency)
- Learned Conv1 filter visualization (weight heatmaps)
- Intermediate activation maps (feature maps across layers)
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


# Set seeds for deterministic comparison
torch.manual_seed(42)
np.random.seed(42)


class FullyConnectedMLP(nn.Module):
    """
    Standard 3-Layer Fully-Connected MLP (discards 2D spatial topology).
    """
    def __init__(self, input_dim=784, hidden1=128, hidden2=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ModernConvNet(nn.Module):
    """
    CS231n Convolutional Architecture:
    [Conv (3x3, 16, pad 1) -> BN -> ReLU -> MaxPool(2x2)]
    -> [Conv (3x3, 32, pad 1) -> BN -> ReLU -> MaxPool(2x2)]
    -> FC(32*7*7 -> 64) -> ReLU -> Dropout -> FC(64 -> 10)
    Total Parameters: ~55,466 (nearly half the parameters of the MLP!)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, test_loader, epochs=6, lr=1e-3, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Test evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        epoch_test_loss = val_loss / total
        epoch_test_acc = (correct / total) * 100.0
        
        history["train_loss"].append(epoch_train_loss)
        history["test_loss"].append(epoch_test_loss)
        history["test_acc"].append(epoch_test_acc)
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")
        
    return history


def main():
    print("=" * 65)
    print("CS231n Lecture 5: ConvNet vs MLP Empirical Benchmark (Fast Local)")
    print("=" * 65)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration Device: {device}")
    
    # 1. Load Local Dataset (ZERO download time)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    full_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform)
    full_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)
    
    # Fast lightweight subset: 6,000 train samples, 1,000 test samples
    train_subset = Subset(full_train, indices=range(6000))
    test_subset = Subset(full_test, indices=range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # 2. Instantiate models
    mlp = FullyConnectedMLP()
    cnn = ModernConvNet()
    
    mlp_params = count_parameters(mlp)
    cnn_params = count_parameters(cnn)
    
    print(f"MLP Parameters:     {mlp_params:,}")
    print(f"ConvNet Parameters: {cnn_params:,} ({cnn_params / mlp_params * 100:.1f}% of MLP size, 2x fewer params!)")
    
    epochs = 6
    print("\n--- Training Fully Connected MLP ---")
    t0 = time.time()
    mlp_hist = train_model(mlp, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    mlp_time = time.time() - t0
    print(f"MLP trained in {mlp_time:.2f}s | Final Test Acc: {mlp_hist['test_acc'][-1]:.2f}%")
    
    print("\n--- Training Modern ConvNet ---")
    t0 = time.time()
    cnn_hist = train_model(cnn, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    cnn_time = time.time() - t0
    print(f"ConvNet trained in {cnn_time:.2f}s | Final Test Acc: {cnn_hist['test_acc'][-1]:.2f}%")
    
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 3. Plot Figure 2: Training & Validation Benchmark
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    
    epochs_range = range(1, epochs + 1)
    
    # Loss curves
    axes[0].plot(epochs_range, mlp_hist["train_loss"], "r--", label="MLP (Train)", linewidth=1.8)
    axes[0].plot(epochs_range, mlp_hist["test_loss"], "r-", label="MLP (Test)", linewidth=2.2)
    axes[0].plot(epochs_range, cnn_hist["train_loss"], "b--", label="ConvNet (Train)", linewidth=1.8)
    axes[0].plot(epochs_range, cnn_hist["test_loss"], "b-", label="ConvNet (Test)", linewidth=2.2)
    axes[0].set_title("Cross-Entropy Loss vs Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(frameon=True)
    
    # Accuracy curves
    axes[1].plot(epochs_range, mlp_hist["test_acc"], "r-o", label=f"MLP ({mlp_hist['test_acc'][-1]:.1f}%)", linewidth=2)
    axes[1].plot(epochs_range, cnn_hist["test_acc"], "b-s", label=f"ConvNet ({cnn_hist['test_acc'][-1]:.1f}%)", linewidth=2.5)
    axes[1].set_title("Test Accuracy (% Correct)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(frameon=True, loc="lower right")
    
    # Parameter Efficiency Bar Chart
    models = ["MLP (FC)", "Modern ConvNet"]
    params = [mlp_params, cnn_params]
    accs = [mlp_hist["test_acc"][-1], cnn_hist["test_acc"][-1]]
    colors = ["#e74c3c", "#2980b9"]
    
    bars = axes[2].bar(models, accs, color=colors, width=0.48, edgecolor="#333333", linewidth=1.2)
    axes[2].set_ylim(60, 95)
    axes[2].set_ylabel("Final Test Accuracy (%)", fontsize=10)
    axes[2].set_title("Parameter Efficiency vs Accuracy", fontsize=11, fontweight="bold")
    for bar, param, acc in zip(bars, params, accs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                     f"{acc:.1f}%\n({param:,} params)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
                     
    fig.suptitle("Stanford CS231n Lecture 5: Fully-Connected MLP vs. ConvNet on Fashion-MNIST (6k samples)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_convnet_vs_mlp_curves.png")
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {fig2_path}")
    
    # 4. Plot Figure 3: Visualizing Learned Conv1 Kernels (Filters)
    conv1_weights = cnn.conv1.weight.data.cpu().numpy()  # (16, 1, 3, 3)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle("Learned 3×3 Filters from Conv1 (16 Kernels)", fontsize=12, fontweight="bold", y=1.05)
    
    for idx in range(16):
        ax = axes[idx // 8, idx % 8]
        kernel = conv1_weights[idx, 0]
        im = ax.imshow(kernel, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(f"Kernel #{idx + 1}", fontsize=9)
        ax.axis("off")
        
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    fig3_path = os.path.join(figures_dir, "03_learned_filters.png")
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved: {fig3_path}")
    
    # 5. Plot Figure 4: Feature Maps (Activation Maps) across Layers
    cnn.eval()
    sample_img, sample_lbl = full_test[18]  # Bag / Shoe
    x_in = sample_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        c1 = cnn.conv1(x_in)
        r1 = cnn.relu1(cnn.bn1(c1))
        p1 = cnn.pool1(r1)
        c2 = cnn.conv2(p1)
        r2 = cnn.relu2(cnn.bn2(c2))
        p2 = cnn.pool2(r2)
        
    f_conv1_np = r1.squeeze(0).cpu().numpy()  # (16, 28, 28)
    f_pool1_np = p1.squeeze(0).cpu().numpy()  # (16, 14, 14)
    f_pool2_np = p2.squeeze(0).cpu().numpy()  # (32, 7, 7)
    
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 9)
    
    # Row 0: Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(sample_img[0], cmap="gray")
    ax_orig.set_title(f"Input: {class_names[sample_lbl]}\n(28 × 28 × 1)", fontweight="bold", fontsize=10)
    ax_orig.axis("off")
    
    # Row 0 (cols 1-8): Conv1 + ReLU feature maps (8 of 16)
    for i in range(8):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(f_conv1_np[i], cmap="viridis")
        ax.set_title(f"Conv1 #{i+1}\n(28×28)", fontsize=8)
        ax.axis("off")
        
    # Row 1 (col 0): Downsampled prompt
    ax_pool_txt = fig.add_subplot(gs[1, 0])
    ax_pool_txt.axis("off")
    ax_pool_txt.text(0.5, 0.5, "Stage 1:\nMaxPool(2×2)\n(14 × 14)", ha="center", va="center",
                     fontweight="bold", bbox=dict(boxstyle="round", facecolor="#eef2f7", edgecolor="#b0c4de"))
    
    # Row 1 (cols 1-8): MaxPool1 outputs
    for i in range(8):
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(f_pool1_np[i], cmap="viridis")
        ax.set_title(f"Pool1 #{i+1}\n(14×14)", fontsize=8)
        ax.axis("off")
        
    # Row 2 (col 0): Stage 2 prompt
    ax_stage2_txt = fig.add_subplot(gs[2, 0])
    ax_stage2_txt.axis("off")
    ax_stage2_txt.text(0.5, 0.5, "Stage 2:\nConv2 + Pool2\n(7 × 7)", ha="center", va="center",
                       fontweight="bold", bbox=dict(boxstyle="round", facecolor="#eef2f7", edgecolor="#b0c4de"))
    
    # Row 2 (cols 1-8): Conv2 + Pool2 feature maps (8 of 32)
    for i in range(8):
        ax = fig.add_subplot(gs[2, i + 1])
        ax.imshow(f_pool2_np[i], cmap="magma")
        ax.set_title(f"Pool2 #{i+1}\n(7×7)", fontsize=8)
        ax.axis("off")
        
    fig.suptitle("Stanford CS231n Lecture 5: Hierarchical Activation Maps (Input → Conv1 → Pool1 → Conv2 → Pool2)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig4_path = os.path.join(figures_dir, "04_feature_maps_and_activations.png")
    plt.savefig(fig4_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved: {fig4_path}")
    print("\nBenchmark and visualizations completed successfully!")


if __name__ == "__main__":
    main()
