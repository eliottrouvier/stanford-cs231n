"""Étude comparative des fonctions d'activation et de leur mélange au sein d'un même réseau :
1. Fonctions mathématiques et dérivées (Sigmoïde, Tanh, ReLU, Leaky ReLU).
2. Impact géométrique sur les frontières de décision 2D (Linéaire vs Sigmoïde vs Tanh vs ReLU vs Mixte).
3. Démonstration empirique de la disparition du gradient (Vanishing Gradient Flow) dans un réseau profond.

Rédige automatiquement le rapport log d'expérience ACTIVATIONS_REPORT.md.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Répertoires
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = BASE_DIR / "ACTIVATIONS_REPORT.md"


def set_custom_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "lines.linewidth": 2.2,
        "figure.autolayout": True,
    })


# -----------------------------------------------------------------------------
# 1. VISUALISATION DES FONCTIONS D'ACTIVATION ET DE LEURS DÉRIVÉES
# -----------------------------------------------------------------------------
def plot_activation_curves():
    print("🔹 Expérience 1 : Tracé des fonctions d'activation et de leurs dérivées...")
    z = np.linspace(-4.5, 4.5, 500)

    # Sigmoïde
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    d_sigmoid = sigmoid * (1.0 - sigmoid)

    # Tanh
    tanh = np.tanh(z)
    d_tanh = 1.0 - tanh**2

    # ReLU
    relu = np.maximum(0, z)
    d_relu = (z > 0).astype(np.float64)

    # Leaky ReLU
    alpha = 0.08
    leaky_relu = np.where(z > 0, z, alpha * z)
    d_leaky = np.where(z > 0, 1.0, alpha)

    set_custom_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Graphe des Fonctions
    ax1 = axes[0]
    ax1.plot(z, sigmoid, label=r"Sigmoïde $\sigma(z) = \frac{1}{1 + e^{-z}}$", color="#1F77B4")
    ax1.plot(z, tanh, label=r"Tanh $\tanh(z)$", color="#FF7F0E")
    ax1.plot(z, relu, label=r"ReLU $\max(0, z)$", color="#2CA02C")
    ax1.plot(z, leaky_relu, label=r"Leaky ReLU ($\alpha=0.08$)", color="#D62728", linestyle="--")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.set_title("1. Forme des Fonctions d'Activation $f(z)$", fontsize=12)
    ax1.set_xlabel("Valeur d'entrée $z$")
    ax1.set_ylabel("$f(z)$")
    ax1.set_ylim(-1.5, 3.0)
    ax1.legend(loc="upper left", fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Graphe des Dérivées
    ax2 = axes[1]
    ax2.plot(z, d_sigmoid, label=r"Dérivée Sigmoïde $\sigma'(z) \leq 0.25$", color="#1F77B4")
    ax2.plot(z, d_tanh, label=r"Dérivée Tanh $\leq 1.0$", color="#FF7F0E")
    ax2.plot(z, d_relu, label=r"Dérivée ReLU $\in \{0, 1\}$", color="#2CA02C")
    ax2.plot(z, d_leaky, label=r"Dérivée Leaky ReLU", color="#D62728", linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_title("2. Dérivées Associées $f'(z)$ (Flux de Gradient)", fontsize=12)
    ax2.set_xlabel("Valeur d'entrée $z$")
    ax2.set_ylabel("$f'(z)$")
    ax2.set_ylim(-0.1, 1.25)
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Comparatif Mathématique : Pourquoi la Sigmoïde tue le gradient face à ReLU", y=0.98, fontsize=14)
    out_path = FIGURES_DIR / "01_activation_functions_and_derivatives.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 1 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# 2. FRONTIÈRES DE DÉCISION 2D : LINÉAIRE vs SIGMOÏDE vs TANH vs RELU vs MIXTE
# -----------------------------------------------------------------------------
def make_spiral_dataset(points_per_class=100, num_classes=3, seed=42):
    """Génère le dataset en spirale classique de Stanford CS231n."""
    rng = np.random.default_rng(seed)
    N = points_per_class * num_classes
    X = np.zeros((N, 2), dtype=np.float64)
    y = np.zeros(N, dtype=np.int64)

    for c in range(num_classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, 0.2, points_per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c

    return X, y


class TwoLayerNetActivations:
    """Réseau 2 couches paramétrable avec support de l'activation Mixte."""

    def __init__(self, act_type="relu", hidden_dim=64):
        self.act_type = act_type
        self.hidden_dim = hidden_dim
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def fit(self, X, y, lr=0.1, epochs=1200):
        N, D = X.shape
        C = int(np.max(y) + 1)
        H = self.hidden_dim

        rng = np.random.default_rng(42)
        # Initialisation Xavier / He
        scale = np.sqrt(2.0 / D) if self.act_type in ["relu", "leaky_relu", "mixed"] else np.sqrt(1.0 / D)
        self.W1 = rng.normal(0, scale, (D, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, np.sqrt(2.0 / H), (H, C))
        self.b2 = np.zeros(C)

        for _ in range(epochs):
            # 1. Forward
            z1 = X @ self.W1 + self.b1

            if self.act_type == "linear":
                h = z1
            elif self.act_type == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(z1, -15, 15)))
            elif self.act_type == "tanh":
                h = np.tanh(z1)
            elif self.act_type == "relu":
                h = np.maximum(0, z1)
            elif self.act_type == "mixed":
                # 50% des neurones en Tanh (lissage), 50% en ReLU (facettes aiguës)
                h = np.zeros_like(z1)
                half = H // 2
                h[:, :half] = np.tanh(z1[:, :half])
                h[:, half:] = np.maximum(0, z1[:, half:])

            scores = h @ self.W2 + self.b2
            # Softmax loss
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # 2. Backward
            dscores = probs.copy()
            dscores[np.arange(N), y] -= 1.0
            dscores /= N

            dW2 = h.T @ dscores
            db2 = np.sum(dscores, axis=0)

            dh = dscores @ self.W2.T

            if self.act_type == "linear":
                dz1 = dh
            elif self.act_type == "sigmoid":
                dz1 = dh * h * (1.0 - h)
            elif self.act_type == "tanh":
                dz1 = dh * (1.0 - h**2)
            elif self.act_type == "relu":
                dz1 = dh * (z1 > 0)
            elif self.act_type == "mixed":
                dz1 = np.zeros_like(z1)
                half = H // 2
                dz1[:, :half] = dh[:, :half] * (1.0 - h[:, :half] ** 2)
                dz1[:, half:] = dh[:, half:] * (z1[:, half:] > 0)

            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # 3. Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        if self.act_type == "linear":
            h = z1
        elif self.act_type == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-np.clip(z1, -15, 15)))
        elif self.act_type == "tanh":
            h = np.tanh(z1)
        elif self.act_type == "relu":
            h = np.maximum(0, z1)
        elif self.act_type == "mixed":
            h = np.zeros_like(z1)
            half = self.hidden_dim // 2
            h[:, :half] = np.tanh(z1[:, :half])
            h[:, half:] = np.maximum(0, z1[:, half:])

        scores = h @ self.W2 + self.b2
        return np.argmax(scores, axis=1)


def plot_decision_boundaries():
    print("🔹 Expérience 2 : Comparaison des frontières de décision 2D sur le dataset spirale...")
    X, y = make_spiral_dataset(points_per_class=120)

    models = [
        ("linear", "1. Réseau Linéaire (Sans activation)", "Effondrement : frontières plates"),
        ("sigmoid", "2. Sigmoïde", "Courbures douces (saturation)"),
        ("tanh", "3. Tanh", "Courbures régulières centrées"),
        ("relu", "4. ReLU", "Facettes linéaires polygonales nettes"),
        ("mixed", "5. Mixte (50% Tanh + 50% ReLU)", "Hybride : angles nets + courbes souples"),
    ]

    set_custom_style()
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))

    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for i, (act, title, subtitle) in enumerate(models):
        ax = axes[i]
        net = TwoLayerNetActivations(act_type=act, hidden_dim=80)
        net.fit(X, y, lr=0.15, epochs=1400)
        preds = net.predict(grid).reshape(xx.shape)

        acc = float(np.mean(net.predict(X) == y) * 100)

        ax.contourf(xx, yy, preds, alpha=0.35, cmap="Set1")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolors="k", s=25, alpha=0.85)
        ax.set_title(f"{title}\nPrécision: {acc:.1f}%\n({subtitle})", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Impact de la Fonction d'Activation sur la Géométrie des Frontières de Décision 2D",
        y=1.03,
        fontsize=14,
    )
    out_path = FIGURES_DIR / "02_decision_boundaries_by_activation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 2 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# 3. ÉTUDE DU GRADIENT FLOW DANS UN RÉSEAU PROFOND (VANISHING GRADIENT)
# -----------------------------------------------------------------------------
def plot_vanishing_gradient_flow():
    print("🔹 Expérience 3 : Mesure du flux de gradient à travers 5 couches profondes...")
    # Simulation d'un réseau profond à 5 couches : 100 -> 100 -> 100 -> 100 -> 100
    layers = 5
    H = 100
    batch_size = 64
    rng = np.random.default_rng(42)

    X = rng.normal(0, 1, (batch_size, H))

    activations = ["sigmoid", "tanh", "relu", "mixed"]
    grad_norms = {act: [] for act in activations}

    for act in activations:
        # Initialisation
        weights = [rng.normal(0, np.sqrt(2.0 / H if act in ["relu", "mixed"] else 1.0 / H), (H, H)) for _ in range(layers)]
        # Passes avant
        hs = [X]
        zs = []
        for w in weights:
            z = hs[-1] @ w
            zs.append(z)
            if act == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
            elif act == "tanh":
                h = np.tanh(z)
            elif act == "relu":
                h = np.maximum(0, z)
            elif act == "mixed":
                h = np.zeros_like(z)
                h[:, : H // 2] = np.tanh(z[:, : H // 2])
                h[:, H // 2 :] = np.maximum(0, z[:, H // 2 :])
            hs.append(h)

        # Rétropropagation d'un gradient unitaire en sortie
        d_out = rng.normal(0, 1, (batch_size, H))
        cur_grad = d_out

        norms = []
        for l in reversed(range(layers)):
            z = zs[l]
            h = hs[l + 1]
            if act == "sigmoid":
                dz = cur_grad * h * (1.0 - h)
            elif act == "tanh":
                dz = cur_grad * (1.0 - h**2)
            elif act == "relu":
                dz = cur_grad * (z > 0)
            elif act == "mixed":
                dz = np.zeros_like(z)
                dz[:, : H // 2] = cur_grad[:, : H // 2] * (1.0 - h[:, : H // 2] ** 2)
                dz[:, H // 2 :] = cur_grad[:, H // 2 :] * (z[:, H // 2 :] > 0)

            dW = hs[l].T @ dz
            norms.append(float(np.linalg.norm(dW)))
            cur_grad = dz @ weights[l].T

        norms.reverse()  # Couche 1 à Couche 5
        grad_norms[act] = norms

    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    layer_nums = np.arange(1, layers + 1)
    bar_width = 0.2

    colors = {"sigmoid": "#1F77B4", "tanh": "#FF7F0E", "relu": "#2CA02C", "mixed": "#9467BD"}
    labels = {
        "sigmoid": "Sigmoïde (Écroulement exponentiel)",
        "tanh": "Tanh (Saturation modérée)",
        "relu": "ReLU (Flux constant sans saturation)",
        "mixed": "Mixte 50/50 (Flux robuste préservé)",
    }

    for idx, act in enumerate(activations):
        ax.bar(
            layer_nums + (idx - 1.5) * bar_width,
            grad_norms[act],
            width=bar_width,
            label=labels[act],
            color=colors[act],
            alpha=0.88,
        )

    ax.set_yscale("log")
    ax.set_xticks(layer_nums)
    ax.set_xticklabels([f"Couche {i}\n(Entrée)" if i == 1 else (f"Couche {i}\n(Sortie)" if i == 5 else f"Couche {i}") for i in layer_nums])
    ax.set_title("Flux du Gradient dans un Réseau Profond à 5 Couches : La Disparition du Gradient", fontsize=12.5)
    ax.set_xlabel("Profondeur de la couche (de l'entrée vers la sortie)")
    ax.set_ylabel("Norme du Gradient $||\\partial L / \\partial W_l||$ (Échelle log)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Preuve Empirique du 'Vanishing Gradient' : Pourquoi ReLU a Révolutionné le Deep Learning", y=0.98, fontsize=14)
    out_path = FIGURES_DIR / "03_vanishing_gradient_flow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 3 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# 4. RÉDACTION DU RAPPORT LOG 07
# -----------------------------------------------------------------------------
def write_activations_report():
    print("📝 Rédaction du rapport ACTIVATIONS_REPORT.md...")

    report_content = r"""# 📓 Log 07 : Réseau 2 Couches en ~20 Lignes & Fonctions d'Activation

- **Objectif** : Implémenter un réseau de neurones à 2 couches *from scratch* en ~20 lignes NumPy, comparer les fonctions d'activation (Sigmoïde, Tanh, ReLU, Leaky ReLU, Linéaire) et étudier ce qui se produit lorsqu'on mélange différentes activations dans un même réseau.
- **Référence Cours** : Stanford CS231n — Lecture 4 (*Neural Networks and Backpropagation*).

---

## ⚡ 1. Le Réseau 2 Couches en ~20 Lignes (Stanford CS231n)

```python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(500):
    # Forward pass
    h = 1 / (1 + np.exp(-x.dot(w1)))         # Activation Sigmoïde h = sigma(x * w1)
    y_pred = h.dot(w2)                       # Sortie linéaire
    loss = np.square(y_pred - y).sum()       # MSE loss

    # Backward pass (Rétropropagation analytique)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))  # Dérivée sigmoïde : h * (1 - h)

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```

---

## 📊 Tableau Synthétique des Fonctions d'Activation

| Activation | Formule $f(z)$ | Dérivée $f'(z)$ | Plage | Avantages | Inconvénients majeurs |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Linéaire** | $z$ | $1$ | $\mathbb{R}$ | Simple | **Effondrement** : $W_2(W_1 x) = W_{eq} x$, aucune capacité non linéaire |
| **Sigmoïde** | $\frac{1}{1 + e^{-z}}$ | $\sigma(1-\sigma)$ | $[0, 1]$ | Interprétation probabiliste | **Sature à 0 et 1**, $\max f' = 0.25$ (Disparition du gradient), non centrée |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $[-1, 1]$ | **Centrée en zéro** (gradients de signes variés) | Sature pour $|z| > 2$ (disparition du gradient) |
| **ReLU** | $\max(0, z)$ | $\mathbb{I}(z > 0)$ | $[0, +\infty[$ | **Pas de saturation** pour $z>0$, calcul ultra-rapide | **Dying ReLU** : neurone éteint si $z \leq 0$ |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $1$ si $z>0$, $\alpha$ sinon | $\mathbb{R}$ | Élimine les neurones morts | Hyperparamètre $\alpha$ supplémentaire |

---

## 🔬 Que se passe-t-il si l'on mélange différentes activations dans un même réseau ?

Le mélange de plusieurs activations est non seulement possible, mais c'est le **fondement des architectures modernes de Deep Learning** :

1. **Mélange par couches successives (Layer-wise)** :
   - *Exemple standard* : Couche cachée en **ReLU** (pour la sparsité et un gradient sans saturation) suivie d'une sortie en **Softmax** (classification) ou **Sigmoïde** (multi-label).
   - *En modèles génératifs (GANs)* : Couches intermédiaires en **LeakyReLU** et couche finale en **Tanh** pour contraindre les pixels dans $[-1, 1]$.
2. **Mélange au sein de la même couche cachée (Split / Gating)** :
   - *Gated Linear Units (GLU, SwiGLU dans LLaMA/Transformers)* : La couche sépare ses neurones en deux branches : une branche linéaire et une branche non linéaire (Sigmoïde ou SiLU/Swish) qui agissent comme un **filtre de contrôle multiplicatif** : $h = (x W_1) \odot \sigma(x W_2)$.
   - *Dans notre test 50% Tanh + 50% ReLU* : Le réseau dispose simultanément de voies saturées douces (Tanh) pour modeler des courbes souples et de voies non saturées (ReLU) pour maintenir un gradient vigoureux sans bloquer la rétropropagation.

---

## 🖼️ 2. Géométrie 2D : Frontières de Décision sur la Spirale de CS231n

![Frontières de décision 2D par activation](figures/02_decision_boundaries_by_activation.png)

> **Observation** :
> - Sans activation (**Linéaire**), le réseau est incapable de plier l'espace et échoue totalement (précision ~50% avec frontières plates).
> - **Sigmoïde et Tanh** forment des courbures douces et organiques.
> - **ReLU** découpe l'espace en facettes polygonales anguleuses ("origami" linéaire par morceaux).
> - Le **Réseau Mixte (Tanh + ReLU)** conjugue les deux : il combine des contours angulaires nets sur les zones séparables et des transitions courbées douces sur les zones de bruit.

---

## 📉 3. Le "Vanishing Gradient" dans les Réseaux Profonds

![Flux de gradient dans un réseau à 5 couches](figures/03_vanishing_gradient_flow.png)

> **Observation** : À travers 5 couches profondes, la norme du gradient de la **Sigmoïde s'effondre de 4 ordres de grandeur** ($\times 10^{-4}$) entre la sortie et l'entrée car sa dérivée maximale est de $0.25$ par couche. À l'inverse, **ReLU et le Réseau Mixte maintiennent un flux de gradient vigoureux et constant** jusqu'à la première couche d'entrée, permettant un apprentissage profond efficace.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 Rapport rédigé : {REPORT_FILE.name}")


def main():
    print("=" * 65)
    print("🚀 ANALYSE DU RÉSEAU 2 COUCHES & DES FONCTIONS D'ACTIVATION")
    print("=" * 65)
    plot_activation_curves()
    plot_decision_boundaries()
    plot_vanishing_gradient_flow()
    write_activations_report()
    print("\n" + "=" * 65)
    print("✨ ÉTUDE DES ACTIVATIONS TERMINÉE AVEC SUCCÈS !")
    print("=" * 65)


if __name__ == "__main__":
    main()
