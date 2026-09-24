"""Implémentation minimale d'un réseau de neurones à 2 couches en ~20 lignes NumPy.
Réplique exacte de la slide du cours Stanford CS231n (Lecture 4).
"""

import numpy as np
from numpy.random import randn

# N: taille du batch (64 exemples)
# D_in: dimension d'entrée (1000 pixels/features)
# H: dimension de la couche cachée (100 neurones)
# D_out: dimension de sortie (10 classes)
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(500):
    # 1. Forward pass (Passe avant)
    h = 1 / (1 + np.exp(-x.dot(w1)))    # Activation sigmoïde : h = sigma(x * w1)
    y_pred = h.dot(w2)                  # Prédiction linéaire : y_pred = h * w2
    loss = np.square(y_pred - y).sum()  # Perte quadratique (MSE)
    if t % 50 == 0 or t == 499:
        print(f"Itération {t:3d} | Perte: {loss:.4f}")

    # 2. Backward pass (Rétropropagation analytique)
    grad_y_pred = 2.0 * (y_pred - y)            # dL / dy_pred
    grad_w2 = h.T.dot(grad_y_pred)              # dL / dw2 = h^T * grad_y_pred
    grad_h = grad_y_pred.dot(w2.T)              # dL / dh = grad_y_pred * w2^T
    grad_w1 = x.T.dot(grad_h * h * (1 - h))     # dL / dw1 = x^T * (grad_h * sigma'(z))

    # 3. Mise à jour des poids (Gradient Descent)
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

print("✨ Entraînement terminé avec succès !")
