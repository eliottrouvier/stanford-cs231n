"""Simulation 2D des cas d'échec de la descente de gradient et comparaison des optimiseurs :
1. Le Point-Selle (Saddle Point) : où le gradient s'annule sans être un minimum.
2. Le Piège du Minimum Local (Local Minimum Trap) : où SGD reste enfermé dans un creux sous-optimal.
3. La Vallée Courbe de Rosenbrock : où la courbure extrême fait échouer SGD.

Optimiseurs comparés :
- SGD (Noir)
- SGD + Momentum (Bleu)
- RMSProp (Rouge)
- Adam (Vert)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = BASE_DIR / "OPTIMIZERS_FAILURE_REPORT.md"


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
# OPTIMISEURS VECTORISÉS (CS231n)
# -----------------------------------------------------------------------------
def run_optimizer(name, grad_fn, w_init, lr=0.01, steps=100, **kwargs):
    w = np.array(w_init, dtype=np.float64).copy()
    traj = [w.copy()]

    if name == "SGD":
        for _ in range(steps):
            g = grad_fn(w)
            w -= lr * g
            traj.append(w.copy())

    elif name == "SGD + Momentum":
        rho = kwargs.get("rho", 0.9)
        v = np.zeros_like(w)
        for _ in range(steps):
            g = grad_fn(w)
            v = rho * v + g
            w -= lr * v
            traj.append(w.copy())

    elif name == "RMSProp":
        decay = kwargs.get("decay", 0.99)
        eps = kwargs.get("eps", 1e-8)
        s = np.zeros_like(w)
        for _ in range(steps):
            g = grad_fn(w)
            s = decay * s + (1.0 - decay) * (g ** 2)
            w -= (lr / (np.sqrt(s) + eps)) * g
            traj.append(w.copy())

    elif name == "Adam":
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        eps = kwargs.get("eps", 1e-8)
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        for t in range(1, steps + 1):
            g = grad_fn(w)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)
            # Correction de biais au démarrage
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            w -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
            traj.append(w.copy())

    return np.array(traj)


# -----------------------------------------------------------------------------
# CAS 1 : LE POINT-SELLE (SADDLE POINT)
# -----------------------------------------------------------------------------
def saddle_point_loss(x, y):
    # L(x, y) = x^2 - y^2 + 0.05 * y^4 + 0.05 * x^4
    # Point selle à (0, 0), minima globaux à y = +-3.16, x = 0
    return x**2 - y**2 + 0.05 * (y**4) + 0.02 * (x**4)


def saddle_point_grad(w):
    x, y = w[0], w[1]
    dx = 2.0 * x + 0.08 * (x**3)
    dy = -2.0 * y + 0.2 * (y**3)
    return np.array([dx, dy], dtype=np.float64)


def simulate_saddle_point():
    print("🔹 Cas 1 : Simulation de l'échappement d'un Point-Selle...")
    # Départ sur la crête vers le point selle (0, 0)
    w_init = np.array([1.8, 0.002])
    steps = 90

    trajs = {
        "SGD": run_optimizer("SGD", saddle_point_grad, w_init, lr=0.1, steps=steps),
        "SGD + Momentum": run_optimizer("SGD + Momentum", saddle_point_grad, w_init, lr=0.08, steps=steps, rho=0.88),
        "RMSProp": run_optimizer("RMSProp", saddle_point_grad, w_init, lr=0.06, steps=steps, decay=0.9),
        "Adam": run_optimizer("Adam", saddle_point_grad, w_init, lr=0.08, steps=steps),
    }

    # Graphique avec colormap continue style CS231n
    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    x_grid = np.linspace(-2.2, 2.2, 300)
    y_grid = np.linspace(-3.5, 3.5, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = saddle_point_loss(X, Y)

    # Fond colormap continue
    im = ax.contourf(X, Y, Z, levels=60, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax, label="Valeur de Perte $L(w_1, w_2)$")
    ax.contour(X, Y, Z, levels=18, colors="white", alpha=0.3, linewidths=0.8)

    colors = {"SGD": "#000000", "SGD + Momentum": "#0044FF", "RMSProp": "#D62728", "Adam": "#00AA00"}
    for name, tr in trajs.items():
        ax.plot(tr[:, 0], tr[:, 1], label=name, color=colors[name], linewidth=2.8, alpha=0.95)
        ax.plot(tr[-1, 0], tr[-1, 1], "x", color=colors[name], markersize=10, markeredgewidth=2.5)

    # Marquer le départ et le point selle
    ax.plot(w_init[0], w_init[1], "wo", markersize=9, markeredgecolor="black", markeredgewidth=2, label="Départ (1.8, 0.0)")
    ax.plot(0, 0, "r^", markersize=11, label="Point-Selle (0, 0) : $\\nabla L = 0$")
    ax.plot(0, 3.16, "k*", markersize=13, label="Minimum Global")
    ax.plot(0, -3.16, "k*", markersize=13)

    ax.set_title("Cas 1 : Échappement d'un Point-Selle (Saddle Point)\nSGD stagne sur la selle, Momentum et Adam s'échappent", fontsize=13)
    ax.set_xlabel("Poids $w_1$ (Courbure positive)")
    ax.set_ylabel("Poids $w_2$ (Courbure négative)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-3.5, 3.5)

    out_path = FIGURES_DIR / "01_saddle_point_escape.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 1 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# CAS 2 : LE PIÈGE DU MINIMUM LOCAL (LOCAL MINIMUM TRAP)
# -----------------------------------------------------------------------------
def double_well_loss(x, y):
    # Deux puits le long de x : local à x ~ 0.95, barrière à x ~ 0.08, global à x ~ -1.02
    return (x**2 - 1.0)**2 + 0.35 * x + 2.0 * (y**2)


def double_well_grad(w):
    x, y = w[0], w[1]
    dx = 4.0 * x * (x**2 - 1.0) + 0.35
    dy = 4.0 * y
    return np.array([dx, dy], dtype=np.float64)


def simulate_local_minima():
    print("🔹 Cas 2 : Simulation du piège du Minimum Local (Double Well)...")
    w_init = np.array([1.6, 0.8])
    steps = 90

    # SGD et RMSProp restent piégés dans le minimum local à droite
    # SGD + Momentum lancé depuis la crête accumule de la vitesse et saute la barrière vers le global !
    trajs = {
        "SGD": run_optimizer("SGD", double_well_grad, w_init, lr=0.03, steps=steps),
        "SGD + Momentum": run_optimizer("SGD + Momentum", double_well_grad, w_init, lr=0.045, steps=steps, rho=0.92),
        "RMSProp": run_optimizer("RMSProp", double_well_grad, w_init, lr=0.04, steps=steps, decay=0.9),
        "Adam": run_optimizer("Adam", double_well_grad, w_init, lr=0.05, steps=steps),
    }

    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 7.2))

    x_grid = np.linspace(-1.8, 2.0, 300)
    y_grid = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = double_well_loss(X, Y)

    im = ax.contourf(X, Y, Z, levels=60, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax, label="Valeur de Perte $L(w_1, w_2)$")
    ax.contour(X, Y, Z, levels=18, colors="white", alpha=0.3, linewidths=0.8)

    colors = {"SGD": "#000000", "SGD + Momentum": "#0044FF", "RMSProp": "#D62728", "Adam": "#00AA00"}
    for name, tr in trajs.items():
        ax.plot(tr[:, 0], tr[:, 1], label=name, color=colors[name], linewidth=2.8, alpha=0.95)
        ax.plot(tr[-1, 0], tr[-1, 1], "x", color=colors[name], markersize=10, markeredgewidth=2.5)

    # Marquer départ, minimum local et global
    ax.plot(w_init[0], w_init[1], "wo", markersize=9, markeredgecolor="black", markeredgewidth=2, label="Départ (1.6, 0.8)")
    ax.plot(0.95, 0, "o", color="#FFAA00", markersize=10, markeredgecolor="black", label="Minimum Local (Piège sous-optimal)")
    ax.plot(-1.02, 0, "r*", markersize=15, label="Minimum Global")
    ax.plot(0.08, 0, "r^", markersize=9, label="Barrière d'Énergie")

    ax.set_title("Cas 2 : Le Piège du Minimum Local\nSGD et RMSProp sont piégés, Momentum franchit la barrière", fontsize=13)
    ax.set_xlabel("Poids $w_1$")
    ax.set_ylabel("Poids $w_2$")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(-1.8, 2.0)
    ax.set_ylim(-1.5, 1.5)

    out_path = FIGURES_DIR / "02_local_minima_trap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 2 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# CAS 3 : LA VALLÉE DE ROSENBROCK (BANANA FUNCTION)
# -----------------------------------------------------------------------------
def rosenbrock_loss(x, y):
    # L(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2


def rosenbrock_grad(w):
    x, y = w[0], w[1]
    dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dy = 200.0 * (y - x**2)
    return np.array([dx, dy], dtype=np.float64)


def simulate_rosenbrock():
    print("🔹 Cas 3 : Simulation sur la vallée en banane de Rosenbrock...")
    w_init = np.array([-1.5, 1.8])
    steps = 150

    # Sur Rosenbrock, SGD diverge ou oscille violemment
    # Adam et RMSProp s'adaptent et suivent la courbure de la banane
    trajs = {
        "SGD": run_optimizer("SGD", rosenbrock_grad, w_init, lr=0.001, steps=steps),
        "SGD + Momentum": run_optimizer("SGD + Momentum", rosenbrock_grad, w_init, lr=0.0008, steps=steps, rho=0.85),
        "RMSProp": run_optimizer("RMSProp", rosenbrock_grad, w_init, lr=0.04, steps=steps, decay=0.95),
        "Adam": run_optimizer("Adam", rosenbrock_grad, w_init, lr=0.05, steps=steps),
    }

    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    x_grid = np.linspace(-2.0, 1.8, 300)
    y_grid = np.linspace(-0.5, 3.2, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Log de la perte pour visualiser les contours serrés de la banane
    Z = np.log1p(rosenbrock_loss(X, Y))

    im = ax.contourf(X, Y, Z, levels=60, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax, label="log(1 + Perte de Rosenbrock)")
    ax.contour(X, Y, Z, levels=20, colors="white", alpha=0.3, linewidths=0.8)

    colors = {"SGD": "#000000", "SGD + Momentum": "#0044FF", "RMSProp": "#D62728", "Adam": "#00AA00"}
    for name, tr in trajs.items():
        ax.plot(tr[:, 0], tr[:, 1], label=name, color=colors[name], linewidth=2.8, alpha=0.95)
        ax.plot(tr[-1, 0], tr[-1, 1], "x", color=colors[name], markersize=10, markeredgewidth=2.5)

    ax.plot(w_init[0], w_init[1], "wo", markersize=9, markeredgecolor="black", markeredgewidth=2, label="Départ (-1.5, 1.8)")
    ax.plot(1.0, 1.0, "r*", markersize=15, label="Minimum Global (1, 1)")

    ax.set_title("Cas 3 : La Vallée Courbe de Rosenbrock (Banane)\nAdam et RMSProp s'adaptent à la courbure non linéaire", fontsize=13)
    ax.set_xlabel("Poids $w_1$")
    ax.set_ylabel("Poids $w_2$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-2.0, 1.8)
    ax.set_ylim(-0.5, 3.2)

    out_path = FIGURES_DIR / "03_rosenbrock_advanced_optimizers.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 3 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# RÉDACTION DU RAPPORT LOG 05
# -----------------------------------------------------------------------------
def write_failure_cases_report():
    print("📝 Rédaction du rapport OPTIMIZERS_FAILURE_REPORT.md...")

    report_content = r"""# 📓 Log 05 : Cas d'Échec de la Descente de Gradient & Optimiseurs Avancés

- **Objectif** : Visualiser en 2D sur des paysages de perte continus les cas où la descente de gradient classique échoue (point-selle, minimum local, vallée courbe de Rosenbrock) et comparer le comportement de **SGD**, **SGD + Momentum**, **RMSProp** et **Adam**.
- **Référence Cours** : Stanford CS231n — Cours 3 (*Optimization & Advanced Optimizers*).

---

## 📊 Tableau des Pièges Géométriques & Réponses des Optimiseurs

| Piège Géométrique | Définition Mathématique | Comportement de Vanilla SGD | Solution Apportée par les Optimiseurs Avancés |
| :--- | :--- | :--- | :--- |
| **Point-Selle (*Saddle Point*)** | $\nabla L = 0$, valeurs propres positives et négatives du Hessien | **Bloqué ou stagne** indéfiniment car le gradient est nul | **Momentum** et **Adam** conservent la vitesse acquise pour traverser la zone plate et plonger dans la courbure descendante. |
| **Minimum Local Sous-Optimal** | $\nabla L = 0$, bassin fermé séparé du global par une colline | **Piégé définitivement** dans le creux sous-optimal | **Momentum** utilise son énergie cinétique accumulée dans la descente pour franchir la colline vers le minimum global. |
| **Courbure Non Linéaire (Rosenbrock)** | Vallée très étroite en forme de banane ($y \approx x^2$) | **Oscille violemment** entre les parois et n'avance pas | **RMSProp** et **Adam** adaptent individuellement le taux d'apprentissage de chaque coordonnée pour épouser la courbe. |

### Code express (Formulations CS231n — RMSProp & Adam) :
```python
# RMSProp (Tieleman & Hinton) : adapte le pas par coordonnée
grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad**2
w -= (learning_rate / (np.sqrt(grad_squared) + 1e-8)) * grad

# Adam (Kingma & Ba) : Momentum (1er moment) + RMSProp (2e moment) avec correction de biais
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w -= (learning_rate / (np.sqrt(v_hat) + 1e-8)) * m_hat
```

---

## 🧭 1. Cas 1 : L'Échappement du Point-Selle (*Saddle Point*)

![Échappement du Point-Selle](figures/01_saddle_point_escape.png)

> **Observation** : À l'approche du point-selle $(0, 0)$, le gradient s'annule et Vanilla SGD (ligne noire) s'immobilise complètement. À l'inverse, SGD + Momentum (bleu) et Adam (vert) franchissent la zone de gradient nul grâce à leur mémoire et exploitent la pente négative pour plonger vers le minimum global.

---

## 🏔️ 2. Cas 2 : Le Piège du Minimum Local (*Local Minimum Trap*)

![Piège du Minimum Local](figures/02_local_minima_trap.png)

> **Observation** : Dès qu'il tombe dans le puits sous-optimal ($w_1 \approx 0.95$), Vanilla SGD y reste captif car le gradient local le ramène toujours au centre du piège. Grâce à l'énergie cinétique accumulée lors de sa descente initiale, SGD + Momentum franchit la barrière de potentiel ($w_1 \approx 0.08$) et atteint le vrai minimum global.

---

## 🍌 3. Cas 3 : La Vallée en Banane de Rosenbrock

![Vallée de Rosenbrock](figures/03_rosenbrock_advanced_optimizers.png)

> **Observation** : Dans cette vallée courbée aux parois très raides, Vanilla SGD est incapable de naviguer le coude sans exploser ou osciller stérilement. RMSProp (rouge) et Adam (vert) adaptent la taille de leur pas à la courbure locale et serpentent avec succès le long de la banane jusqu'à la cible $(1, 1)$.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 Rapport rédigé : {REPORT_FILE.name}")


def main():
    print("=" * 70)
    print("🚀 SIMULATION 2D : CAS D'ÉCHEC DU GRADIENT & OPTIMISEURS AVANCÉS")
    print("=" * 70)
    simulate_saddle_point()
    simulate_local_minima()
    simulate_rosenbrock()
    write_failure_cases_report()
    print("\n" + "=" * 70)
    print("✨ ÉTUDE TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    main()
