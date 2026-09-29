"""
architecture_zoo_comparison.py - CS231n Lecture 9: CNN Architectures Deep-Dive & Zoo Comparison
------------------------------------------------------------------------------------------------
1. Mathematical Receptive Field Analysis (VGG Insight):
   - Demonstrates why stacking two 3x3 convs is strictly superior to one 5x5 conv (28% fewer parameters + 2 non-linearities).
   - Three 3x3 convs vs one 7x7 conv (45% fewer parameters + 3 non-linearities).
2. Landmark ImageNet Architectures Comparison:
   - AlexNet (2012) -> VGG-16 (2014) -> GoogLeNet Inception (2014) -> ResNet-50 (2015) -> MobileNet-v2 (2018).
   - Metrics: Parameter count (M), GFLOPs, Memory Footprint, and ImageNet Top-1 Accuracy.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


os.makedirs("10_cnn_architectures/figures", exist_ok=True)


def analyze_receptive_field_and_parameters():
    print("--- 1. Analyse Mathématique du Receptive Field (VGG Philosophy) ---")
    C = 64 # Exemple avec 64 canaux
    
    # 1 conv 5x5 vs 2 convs 3x3
    params_5x5 = 1 * (C * C * 5 * 5)
    params_two_3x3 = 2 * (C * C * 3 * 3)
    reduction_5x5 = (1.0 - params_two_3x3 / params_5x5) * 100.0
    
    print(f"Champ Récepteur 5x5 :")
    print(f"  - 1x Conv 5x5 : {params_5x5:,} paramètres (1 non-linéarité)")
    print(f"  - 2x Conv 3x3 : {params_two_3x3:,} paramètres (2 non-linéarités)")
    print(f"  -> Réduction de paramètres : {reduction_5x5:.1f}%")
    
    # 1 conv 7x7 vs 3 convs 3x3
    params_7x7 = 1 * (C * C * 7 * 7)
    params_three_3x3 = 3 * (C * C * 3 * 3)
    reduction_7x7 = (1.0 - params_three_3x3 / params_7x7) * 100.0
    
    print(f"\nChamp Récepteur 7x7 :")
    print(f"  - 1x Conv 7x7 : {params_7x7:,} paramètres (1 non-linéarité)")
    print(f"  - 3x Conv 3x3 : {params_three_3x3:,} paramètres (3 non-linéarités)")
    print(f"  -> Réduction de paramètres : {reduction_7x7:.1f}%")


def plot_architecture_zoo():
    print("\n--- 2. Génération du Comparatif de l'Architecture Zoo ---")
    
    models = [
        {"name": "AlexNet (2012)", "year": 2012, "top1": 63.3, "params": 61.1, "gflops": 0.72, "color": "#94a3b8"},
        {"name": "VGG-16 (2014)", "year": 2014, "top1": 71.5, "params": 138.4, "gflops": 15.5, "color": "#f59e0b"},
        {"name": "GoogLeNet (2014)", "year": 2014, "top1": 69.8, "params": 6.8, "gflops": 1.5, "color": "#8b5cf6"},
        {"name": "ResNet-18 (2015)", "year": 2015, "top1": 69.8, "params": 11.7, "gflops": 1.8, "color": "#38bdf8"},
        {"name": "ResNet-50 (2015)", "year": 2015, "top1": 76.1, "params": 25.6, "gflops": 4.1, "color": "#0284c7"},
        {"name": "ResNet-152 (2015)", "year": 2015, "top1": 78.3, "params": 60.2, "gflops": 11.3, "color": "#0369a1"},
        {"name": "MobileNet-V2 (2018)", "year": 2018, "top1": 72.0, "params": 3.5, "gflops": 0.31, "color": "#10b981"},
        {"name": "ConvNeXt-T (2022)", "year": 2022, "top1": 82.1, "params": 28.6, "gflops": 4.5, "color": "#ec4899"}
    ]
    
    names = [m["name"] for m in models]
    top1s = [m["top1"] for m in models]
    params = [m["params"] for m in models]
    gflops = [m["gflops"] for m in models]
    colors = [m["color"] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("CS231n Lecture 9 : Panorama des Architectures CNN Clés (ImageNet)", fontsize=15, fontweight="bold")
    
    # 1. ImageNet Top-1 Accuracy
    bars1 = axes[0].barh(names, top1s, color=colors, edgecolor="#334155", height=0.65)
    axes[0].set_title("Précision ImageNet Top-1 (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlim(55, 85)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
        
    # 2. Number of Parameters (M)
    bars2 = axes[1].barh(names, params, color=colors, edgecolor="#334155", height=0.65)
    axes[1].set_title("Nombre de Paramètres (Millions)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    axes[1].set_yticks([])
    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}M", va="center", fontsize=9, fontweight="bold")
        
    # 3. Operations (GFLOPs)
    bars3 = axes[2].barh(names, gflops, color=colors, edgecolor="#334155", height=0.65)
    axes[2].set_title("Coût de Calcul (GFLOPs / Forward)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)
    axes[2].set_yticks([])
    for bar in bars3:
        w = bar.get_width()
        axes[2].text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}G", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    save_path = "10_cnn_architectures/figures/architecture_zoo_comparison.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Graphique sauvegardé : {save_path}")


if __name__ == "__main__":
    analyze_receptive_field_and_parameters()
    plot_architecture_zoo()
