"""
CS231n Lecture 5: Convolution Mechanics & Arithmetic from Scratch (NumPy)
--------------------------------------------------------------------------
Implements 2D convolution and Max-Pooling forward passes in pure NumPy.
Verifies the spatial dimension formula: Output = (Input - Filter + 2*Pad)/Stride + 1.
Demonstrates classical edge-detection kernels (Sobel, Laplacian) on real image data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


def conv2d_forward_numpy(x, w, b, stride=1, pad=0):
    """
    Pure NumPy forward pass for a 2D convolution layer.
    
    Arguments:
        x: Input volume of shape (N, C_in, H_in, W_in)
        w: Filter weights of shape (C_out, C_in, KH, KW)
        b: Biases of shape (C_out,)
        stride: Stride step size S
        pad: Zero-padding size P
        
    Returns:
        out: Output volume of shape (N, C_out, H_out, W_out)
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = w.shape
    
    # Stanford CS231n formula: Output = floor((W - F + 2P) / S) + 1
    H_out = int((H_in - KH + 2 * pad) / stride) + 1
    W_out = int((W_in - KW + 2 * pad) / stride) + 1
    
    # Zero-padding
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )
    
    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
    
    # Spatial convolution
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + KH
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + KW
                    
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, c_out, i, j] = np.sum(window * w[c_out]) + b[c_out]
                    
    return out


def max_pool2d_forward_numpy(x, pool_size=2, stride=2):
    """
    Pure NumPy forward pass for 2D Max-Pooling.
    """
    N, C, H, W = x.shape
    H_out = int((H - pool_size) / stride) + 1
    W_out = int((W - pool_size) / stride) + 1
    
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + pool_size
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + pool_size
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(window)
    return out


def main():
    print("=" * 60)
    print("CS231n Lecture 5: Convolution Mechanics (NumPy from Scratch)")
    print("=" * 60)
    
    # 1. Load a single Fashion-MNIST sample
    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    sample_img, label = dataset[0]  # Ankle boot (1, 28, 28)
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    x = sample_img.unsqueeze(0).numpy()  # (1, 1, 28, 28)
    
    print(f"Sample input shape: {x.shape} (Class: {class_names[label]})")
    
    # 2. Handcrafted filter kernels (3x3)
    kernels = {
        "Vertical Edge (Sobel-X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "Horizontal Edge (Sobel-Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "Outline (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    }
    
    # 3. Stack filters into a weight tensor (K, 1, 3, 3)
    k_names = list(kernels.keys())
    W_stack = np.stack([kernels[k] for k in k_names], axis=0)[:, np.newaxis, :, :]  # (5, 1, 3, 3)
    b_zero = np.zeros(len(k_names), dtype=np.float32)
    
    # Apply conv with pad=1, stride=1 (preserves 28x28)
    out_conv = conv2d_forward_numpy(x, W_stack, b_zero, stride=1, pad=1)
    # Apply max-pool (2x2, stride=2 -> 14x14)
    out_pooled = max_pool2d_forward_numpy(out_conv, pool_size=2, stride=2)
    
    print(f"Conv2D output shape: {out_conv.shape} (Input: 28x28, Pad: 1, Stride: 1 -> 28x28)")
    print(f"MaxPool2D output shape: {out_pooled.shape} (Pool: 2x2, Stride: 2 -> 14x14)")
    
    # 4. Generate Publication-Quality Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    
    # Original Image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(x[0, 0], cmap="gray")
    ax_orig.set_title(f"Input: {class_names[label]}\n(28 × 28 × 1)", fontsize=11, fontweight="bold")
    ax_orig.axis("off")
    
    # Filter outputs (Conv)
    for idx, name in enumerate(k_names):
        ax = fig.add_subplot(gs[0, idx + 1])
        ax.imshow(out_conv[0, idx], cmap="coolwarm")
        ax.set_title(f"Conv: {name}\nPad=1, S=1 → (28×28)", fontsize=9, fontweight="bold")
        ax.axis("off")
        
    # Pooled outputs
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")
    ax_empty.text(0.5, 0.5, "Max-Pooling\n(2×2, Stride 2)\nDownsampled\n(14 × 14)",
                  ha="center", va="center", fontsize=11, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999999"))
    
    for idx, name in enumerate(k_names):
        ax = fig.add_subplot(gs[1, idx + 1])
        ax.imshow(out_pooled[0, idx], cmap="coolwarm")
        ax.set_title(f"Pooled: {name}\nDownsampled (14×14)", fontsize=9, fontweight="bold")
        ax.axis("off")
        
    fig.suptitle("Stanford CS231n Lecture 5: Spatial Filtering & Max-Pooling Mechanics (NumPy)",
                 fontsize=14, fontweight="bold", y=1.02)
    
    out_dir = os.path.dirname(__file__)
    fig_path = os.path.join(out_dir, "figures", "01_conv_arithmetic_and_kernels.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved successfully to: {fig_path}")


if __name__ == "__main__":
    main()
