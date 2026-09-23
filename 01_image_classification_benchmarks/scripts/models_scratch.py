"""Implémentation 'from scratch' en NumPy des algorithmes fondamentaux de CS231n :
- k-Nearest Neighbor (kNN) avec calcul matriciel sans boucle
- Linear SVM (Multiclass Hinge Loss) avec gradient analytique vectorisé
- Softmax Classifier (Cross-Entropy Loss) avec gradient analytique vectorisé
"""

import numpy as np


class KNearestNeighbor:
    """Classifieur k-Nearest Neighbors implémenté from scratch avec calcul vectorisé."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbor":
        """Mémorise les données d'entraînement (temps d'entraînement O(1))."""
        self.X_train = X
        self.y_train = y
        return self

    def compute_distances(self, X: np.ndarray, batch_size: int = 500) -> np.ndarray:
        """Calcule la matrice de distances Euclidiennes L2 sans boucle grâce à l'identité :
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>.

        Le calcul s'effectue par mini-lots pour éviter les pics de mémoire vive.
        """
        assert self.X_train is not None, "Le modèle doit être entraîné via fit() d'abord."
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=np.float32)

        train_sq = np.sum(self.X_train ** 2, axis=1)  # (N_train,)

        for start_idx in range(0, num_test, batch_size):
            end_idx = min(start_idx + batch_size, num_test)
            X_batch = X[start_idx:end_idx]
            test_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)  # (batch, 1)
            dot_product = np.dot(X_batch, self.X_train.T)  # (batch, N_train)

            # dist^2 = x^2 + y^2 - 2xy
            dists_sq = test_sq + train_sq - 2 * dot_product
            # Écrêter à 0 pour éviter les infimes valeurs négatives dues à la précision flottante
            dists[start_idx:end_idx] = np.sqrt(np.maximum(dists_sq, 0.0))

        return dists

    def predict(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """Prédit les étiquettes en trouvant les k voisins les plus proches et en votant majoritairement."""
        assert self.y_train is not None
        k_val = k if k is not None else self.k
        dists = self.compute_distances(X)
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=np.int64)

        for i in range(num_test):
            # Trouver les indices des k plus petites distances
            closest_y = self.y_train[np.argpartition(dists[i], k_val)[:k_val]]
            # Vote majoritaire
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred


class LinearSVMScratch:
    """Classifieur linéaire avec fonction de perte Multiclass SVM (Hinge Loss)
    et régularisation L2, optimisé par Mini-Batch SGD.
    """

    def __init__(self, reg: float = 1e-4, delta: float = 1.0):
        self.reg = reg
        self.delta = delta
        self.W: np.ndarray | None = None  # (C, D)
        self.b: np.ndarray | None = None  # (C,)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-2,
        batch_size: int = 128,
        epochs: int = 25,
        verbose: bool = False,
    ) -> list[float]:
        """Entraîne les poids W et le biais b par descente de gradient stochastique."""
        num_samples, dim = X.shape
        num_classes = int(np.max(y) + 1)

        # Initialisation aléatoire gaussienne à petite échelle
        rng = np.random.default_rng(42)
        self.W = rng.normal(0.0, 0.01, size=(num_classes, dim)).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

        loss_history = []

        for epoch in range(epochs):
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                B = X_batch.shape[0]

                # Scores : s = X W^T + b  -> (B, C)
                scores = np.dot(X_batch, self.W.T) + self.b

                # Scores de la classe correcte pour chaque échantillon : s_{y_i}
                correct_scores = scores[np.arange(B), y_batch][:, np.newaxis]

                # Marges SVM : max(0, s_j - s_{y_i} + delta)
                margins = np.maximum(0, scores - correct_scores + self.delta)
                margins[np.arange(B), y_batch] = 0.0  # ne pas compter j = y_i

                # Perte batch + régularisation L2
                data_loss = np.sum(margins) / B
                reg_loss = 0.5 * self.reg * np.sum(self.W ** 2)
                loss = data_loss + reg_loss
                epoch_losses.append(loss)

                # Gradient analytique
                # Matrice binaire indiquant les violations de marge
                margin_mask = (margins > 0).astype(np.float32)
                # Pour la classe correcte, le gradient est la somme négative des violations
                correct_counts = np.sum(margin_mask, axis=1)
                margin_mask[np.arange(B), y_batch] = -correct_counts

                # dW = (margin_mask^T @ X) / B + reg * W  -> (C, D)
                dW = np.dot(margin_mask.T, X_batch) / B + self.reg * self.W
                db = np.sum(margin_mask, axis=0) / B

                # Mise à jour des paramètres
                self.W -= lr * dW
                self.b -= lr * db

            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"SVM Epoch [{epoch+1}/{epochs}] - Perte: {avg_loss:.4f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit la classe ayant le score maximal."""
        assert self.W is not None and self.b is not None
        scores = np.dot(X, self.W.T) + self.b
        return np.argmax(scores, axis=1)


class SoftmaxClassifierScratch:
    """Classifieur Softmax (Régression logistique multinomiale / Cross-Entropy)
    avec régularisation L2, optimisé par Mini-Batch SGD avec momentum.
    """

    def __init__(self, reg: float = 1e-4):
        self.reg = reg
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 5e-2,
        momentum: float = 0.9,
        batch_size: int = 128,
        epochs: int = 25,
        verbose: bool = False,
    ) -> list[float]:
        num_samples, dim = X.shape
        num_classes = int(np.max(y) + 1)

        rng = np.random.default_rng(42)
        self.W = rng.normal(0.0, 0.01, size=(num_classes, dim)).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

        v_W = np.zeros_like(self.W)
        v_b = np.zeros_like(self.b)

        loss_history = []

        for epoch in range(epochs):
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                B = X_batch.shape[0]

                # Scores stables numériquement : s = X W^T + b
                scores = np.dot(X_batch, self.W.T) + self.b
                # Stabilité numérique : soustraire le max par ligne
                scores -= np.max(scores, axis=1, keepdims=True)

                exp_scores = np.exp(scores)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (B, C)

                # Cross-Entropy Loss : -log P(y_i)
                correct_log_probs = -np.log(np.maximum(probs[np.arange(B), y_batch], 1e-12))
                data_loss = np.sum(correct_log_probs) / B
                reg_loss = 0.5 * self.reg * np.sum(self.W ** 2)
                loss = data_loss + reg_loss
                epoch_losses.append(loss)

                # Gradient analytique : dL/ds = P - 1_{y_i}
                dscores = probs.copy()
                dscores[np.arange(B), y_batch] -= 1.0
                dscores /= B

                dW = np.dot(dscores.T, X_batch) + self.reg * self.W
                db = np.sum(dscores, axis=0)

                # SGD avec Momentum
                v_W = momentum * v_W - lr * dW
                v_b = momentum * v_b - lr * db

                self.W += v_W
                self.b += v_b

            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Softmax Epoch [{epoch+1}/{epochs}] - Perte: {avg_loss:.4f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        scores = np.dot(X, self.W.T) + self.b
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        scores = np.dot(X, self.W.T) + self.b
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
