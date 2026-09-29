"""
rnn_vs_lstm_vanishing_gradient.py - CS231n Lecture 10: Vanishing Gradients & LSTM Highway
-----------------------------------------------------------------------------------------
Quantifies & visualizes:
1. The Vanishing / Exploding Gradient problem in Vanilla RNN vs. LSTM vs. GRU.
   - Measures ||dL / dh_0|| as sequence length T increases from 5 to 50 steps.
   - Shows Vanilla RNN gradients vanishing exponentially towards 0.
   - Shows LSTM maintaining steady error backpropagation through its additive cell state carousel.
2. The role of Gradient Clipping (Pascanu et al., 2013) to cap exploding gradient norms.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Chapter 11 - Gradients] Running on device: {device}")

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("11_recurrent_neural_networks/figures", exist_ok=True)


def measure_gradient_flow_over_time():
    print("\n--- Analyse du Flux de Gradient : Vanilla RNN vs LSTM vs GRU ---")
    sequence_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    hidden_dim = 64
    input_dim = 32
    batch_size = 16

    architectures = [
        ("Vanilla RNN (tanh)", nn.RNN, "#ef4444", "-"),
        ("GRU (Chung et al., 2014)", nn.GRU, "#f59e0b", "-."),
        ("LSTM (Hochreiter & Schmidhuber)", nn.LSTM, "#0284c7", "-")
    ]

    results = {name: [] for name, _, _, _ in architectures}

    for T in sequence_lengths:
        for name, rnn_class, _, _ in architectures:
            # Create recurrent module
            module = rnn_class(input_dim, hidden_dim, batch_first=True).to(device)
            # Dummy sequential input
            x = torch.randn(batch_size, T, input_dim, device=device)
            
            # Initial hidden state requiring gradient
            if name == "LSTM (Hochreiter & Schmidhuber)":
                h0 = torch.zeros(1, batch_size, hidden_dim, device=device, requires_grad=True)
                c0 = torch.zeros(1, batch_size, hidden_dim, device=device)
                out, (hn, cn) = module(x, (h0, c0))
            else:
                h0 = torch.zeros(1, batch_size, hidden_dim, device=device, requires_grad=True)
                out, hn = module(x, h0)

            # Target loss based on the final time-step output
            target = torch.randn_like(out[:, -1, :])
            loss = nn.MSELoss()(out[:, -1, :], target)
            
            # Backpropagate through time (BPTT)
            loss.backward()
            
            # Compute norm of gradient arriving at the initial hidden state h_0
            grad_norm = h0.grad.norm().item()
            results[name].append(grad_norm)

        print(f"Longueur T={T:02d} | RNN ||dL/dh0||={results['Vanilla RNN (tanh)'][-1]:.2e} | LSTM ||dL/dh0||={results['LSTM (Hochreiter & Schmidhuber)'][-1]:.2e}")

    # Plot gradient norms vs sequence length (Log scale)
    plt.figure(figsize=(12, 5.5))
    
    for name, _, color, style in architectures:
        plt.plot(sequence_lengths, results[name], label=name, color=color, linestyle=style, linewidth=2.5, marker="o")

    plt.yscale("log")
    plt.title("CS231n Lecture 10 : Évanouissement du Gradient selon la Longueur de la Séquence T", fontsize=13, fontweight="bold")
    plt.xlabel("Longueur de la Séquence (Pas de Temps T)", fontsize=11)
    plt.ylabel(r"Norme du Gradient $\| \nabla_{h_0} \mathcal{L} \|$ (Échelle Logarithmique)", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.axhline(1e-4, color="#94a3b8", linestyle=":", label="Seuil d'Évanouissement (< 1e-4)")
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10.5)
    
    plt.tight_layout()
    save_path = "11_recurrent_neural_networks/figures/vanishing_gradient_rnn_vs_lstm.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n-> Graphique de flux de gradient sauvegardé : {save_path}")


if __name__ == "__main__":
    t0 = time.time()
    measure_gradient_flow_over_time()
    print(f"=== Analyse du gradient terminée en {time.time() - t0:.2f}s ===")
