"""Simulation approfondie des rouages d'Adam (Adaptive Moment Estimation) :
1. Décomposition de la synergie : Momentum (1er moment) + RMSProp (2nd moment) = Adam.
2. Démonstration du rôle vital de la correction de biais au démarrage.

Génère les visualisations 2D et rédige le rapport d'expérience ADAM_REPORT.md.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Répertoires
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = BASE_DIR / "ADAM_REPORT.md"


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


def loss_and_grad(w):
    """Fonction de ravin mal conditionné : L(w1, w2) = 0.5 * (0.06 * w1^2 + 1.6 * w2^2).
    La courbure le long de w2 est ~27x plus forte que le long de w1.
    """
    w1, w2 = w[0], w[1]
    loss = 0.5 * (0.06 * (w1**2) + 1.6 * (w2**2))
    grad = np.array([0.06 * w1, 1.6 * w2], dtype=np.float64)
    return float(loss), grad


# -----------------------------------------------------------------------------
# EXPÉRIENCE 1 : LA SYNTHÈSE (MOMENTUM vs RMSPROP vs ADAM)
# -----------------------------------------------------------------------------
def run_synthesis_experiment():
    print("🔹 Expérience 1 : Synergie Momentum + RMSProp = Adam...")
    w_init = np.array([-4.0, 1.8], dtype=np.float64)
    steps = 60

    # 1. Momentum Seul (beta1 = 0.9)
    # Gère l'inertie mais ne normalise pas les échelles différentes
    w_mom = w_init.copy()
    v_mom = np.zeros_like(w_mom)
    traj_mom = [w_mom.copy()]
    loss_mom = []
    lr_mom = 0.6
    for _ in range(steps):
        l, g = loss_and_grad(w_mom)
        loss_mom.append(l)
        v_mom = 0.88 * v_mom + g
        w_mom -= lr_mom * v_mom
        traj_mom.append(w_mom.copy())
    loss_mom.append(loss_and_grad(w_mom)[0])

    # 2. RMSProp Seul (beta2 = 0.99)
    # Met à l'échelle les coordonnées mais manque de vélocité
    w_rms = w_init.copy()
    s_rms = np.zeros_like(w_rms)
    traj_rms = [w_rms.copy()]
    loss_rms = []
    lr_rms = 0.15
    for _ in range(steps):
        l, g = loss_and_grad(w_rms)
        loss_rms.append(l)
        s_rms = 0.95 * s_rms + 0.05 * (g**2)
        w_rms -= (lr_rms / (np.sqrt(s_rms) + 1e-8)) * g
        traj_rms.append(w_rms.copy())
    loss_rms.append(loss_and_grad(w_rms)[0])

    # 3. Adam Complet (beta1 = 0.9, beta2 = 0.999 avec correction de biais)
    w_adam = w_init.copy()
    m_adam = np.zeros_like(w_adam)
    v_adam = np.zeros_like(w_adam)
    traj_adam = [w_adam.copy()]
    loss_adam = []
    lr_adam = 0.25
    b1, b2 = 0.9, 0.999
    for t in range(1, steps + 1):
        l, g = loss_and_grad(w_adam)
        loss_adam.append(l)
        m_adam = b1 * m_adam + (1.0 - b1) * g
        v_adam = b2 * v_adam + (1.0 - b2) * (g**2)
        m_hat = m_adam / (1.0 - b1**t)
        v_hat = v_adam / (1.0 - b2**t)
        w_adam -= (lr_adam / (np.sqrt(v_hat) + 1e-8)) * m_hat
        traj_adam.append(w_adam.copy())
    loss_adam.append(loss_and_grad(w_adam)[0])

    trajectories = {
        "Momentum Seul": {"traj": np.array(traj_mom), "loss": loss_mom, "color": "#0044FF", "style": "-"},
        "RMSProp Seul": {"traj": np.array(traj_rms), "loss": loss_rms, "color": "#D62728", "style": "-"},
        "Adam (Momentum + RMSProp)": {"traj": np.array(traj_adam), "loss": loss_adam, "color": "#00AA00", "style": "-"},
    }

    # Tracé Figure 1
    set_custom_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1 : Trajectoires 2D
    ax = axes[0]
    w1_grid = np.linspace(-4.5, 1.0, 300)
    w2_grid = np.linspace(-2.2, 2.2, 300)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = 0.5 * (0.06 * W1**2 + 1.6 * W2**2)

    im = ax.contourf(W1, W2, Z, levels=50, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax, label="Perte $L(w_1, w_2)$")
    ax.contour(W1, W2, Z, levels=18, colors="white", alpha=0.3, linewidths=0.8)

    for name, data in trajectories.items():
        tr = data["traj"]
        ax.plot(tr[:, 0], tr[:, 1], data["style"], label=name, color=data["color"], linewidth=2.8, alpha=0.95)
        ax.plot(tr[-1, 0], tr[-1, 1], "x", color=data["color"], markersize=10, markeredgewidth=2.5)

    ax.plot(w_init[0], w_init[1], "wo", markersize=9, markeredgecolor="black", markeredgewidth=2, label="Départ (-4.0, 1.8)")
    ax.plot(0, 0, "r*", markersize=14, label="Minimum (0, 0)")
    ax.set_title("1. Trajectoires 2D : Pourquoi fusionner Momentum & RMSProp", fontsize=12)
    ax.set_xlabel(r"Axe plat $w_1$")
    ax.set_ylabel(r"Axe raide $w_2$")
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel 2 : Courbes de convergence
    ax2 = axes[1]
    for name, data in trajectories.items():
        ax2.plot(data["loss"], label=name, color=data["color"], linewidth=2.5, alpha=0.9)

    ax2.set_yscale("log")
    ax2.set_title("2. Vitesse de Convergence : Perte vs Nombre de pas", fontsize=12)
    ax2.set_xlabel("Nombre d'itérations (Steps)")
    ax2.set_ylabel("Perte $L(W)$ (Échelle log)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("La Synergie d'Adam : Vitesse d'Inertie (Momentum) + Normalisation d'Échelle (RMSProp)", y=0.98, fontsize=14)
    out_path = FIGURES_DIR / "01_adam_synthesis_2d.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 1 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# EXPÉRIENCE 2 : LE RÔLE DE LA CORRECTION DE BIAIS (BIAS CORRECTION)
# -----------------------------------------------------------------------------
def run_bias_correction_experiment():
    print("🔹 Expérience 2 : L'impact de la correction de biais au démarrage...")
    w_init = np.array([-3.5, 1.5], dtype=np.float64)
    steps = 30
    lr = 0.25
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    # 1. Adam AVEC correction de biais
    w_corr = w_init.copy()
    m_c = np.zeros_like(w_corr)
    v_c = np.zeros_like(w_corr)
    traj_corr = [w_corr.copy()]
    eff_lr_corr = []

    for t in range(1, steps + 1):
        _, g = loss_and_grad(w_corr)
        m_c = b1 * m_c + (1.0 - b1) * g
        v_c = b2 * v_c + (1.0 - b2) * (g**2)
        m_hat = m_c / (1.0 - b1**t)
        v_hat = v_c / (1.0 - b2**t)
        step_vec = (lr / (np.sqrt(v_hat) + eps)) * m_hat
        eff_lr_corr.append(np.linalg.norm(step_vec))
        w_corr -= step_vec
        traj_corr.append(w_corr.copy())

    # 2. Adam SANS correction de biais (Initialisation biaisée vers zéro)
    # À t=1, v1 ~ 0.001 * g^2, donc sqrt(v1) ~ 0.0316 * |g|
    # m1 ~ 0.1 * g, donc le premier pas est ~ 0.1 / 0.0316 * lr = 3.16 * lr !
    w_nocorr = w_init.copy()
    m_nc = np.zeros_like(w_nocorr)
    v_nc = np.zeros_like(w_nocorr)
    traj_nocorr = [w_nocorr.copy()]
    eff_lr_nocorr = []

    for t in range(1, steps + 1):
        _, g = loss_and_grad(w_nocorr)
        m_nc = b1 * m_nc + (1.0 - b1) * g
        v_nc = b2 * v_nc + (1.0 - b2) * (g**2)
        # Pas de division par (1 - beta^t) !
        step_vec = (lr / (np.sqrt(v_nc) + eps)) * m_nc
        eff_lr_nocorr.append(np.linalg.norm(step_vec))
        w_nocorr -= step_vec
        traj_nocorr.append(w_nocorr.copy())

    traj_corr = np.array(traj_corr)
    traj_nocorr = np.array(traj_nocorr)

    set_custom_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1 : Trajectoires des 25 premiers pas
    ax1 = axes[0]
    w1_grid = np.linspace(-4.5, 2.0, 300)
    w2_grid = np.linspace(-3.0, 3.0, 300)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = 0.5 * (0.06 * W1**2 + 1.6 * W2**2)

    im = ax1.contourf(W1, W2, Z, levels=50, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax1, label="Perte $L(w_1, w_2)$")
    ax1.contour(W1, W2, Z, levels=18, colors="white", alpha=0.3, linewidths=0.8)

    ax1.plot(traj_corr[:, 0], traj_corr[:, 1], "o-", color="#00AA00", label="Adam AVEC Correction de Biais", linewidth=2.8, markersize=5)
    ax1.plot(traj_nocorr[:, 0], traj_nocorr[:, 1], "s--", color="#FF3300", label="Adam SANS Correction (Explosion initiale)", linewidth=2.5, markersize=5)

    ax1.plot(w_init[0], w_init[1], "wo", markersize=9, markeredgecolor="black", markeredgewidth=2, label="Départ (-3.5, 1.5)")
    ax1.plot(0, 0, "r*", markersize=14, label="Minimum (0, 0)")
    ax1.set_title("1. Premiers pas : Stabilité vs Sursaut Catastrophique", fontsize=12)
    ax1.set_xlabel(r"Poids $w_1$")
    ax1.set_ylabel(r"Poids $w_2$")
    ax1.legend(loc="lower left", framealpha=0.9)

    # Panel 2 : Norme de la mise à jour (taille du pas)
    ax2 = axes[1]
    ax2.plot(range(1, steps + 1), eff_lr_nocorr, "s--", color="#FF3300", label="Sans Correction : Pas initial démesuré", linewidth=2.5)
    ax2.plot(range(1, steps + 1), eff_lr_corr, "o-", color="#00AA00", label="Avec Correction : Pas régulier et amorti", linewidth=2.5)

    ax2.set_title("2. Amplitude de la mise à jour ||Δw|| au démarrage", fontsize=12)
    ax2.set_xlabel("Itération $t$")
    ax2.set_ylabel("Norme du pas $||\\Delta w_t||$")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Pourquoi la Correction de Biais est Vitale dans Adam", y=0.98, fontsize=14)
    out_path = FIGURES_DIR / "02_bias_correction_impact.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 2 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# EXPÉRIENCE 3 : RÉPLIQUE EXACTE DE LA SLIDE DE COURS CS231N
# -----------------------------------------------------------------------------
def run_cs231n_slide_replica():
    print("🔹 Expérience 3 : Réplique exacte de la slide CS231n (Bol tourné 2D)...")
    theta = np.radians(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def rotated_loss_and_grad(w):
        w_rot = R @ w
        loss = 0.5 * (0.04 * (w_rot[0]**2) + 1.1 * (w_rot[1]**2))
        grad_rot = np.array([0.04 * w_rot[0], 1.1 * w_rot[1]])
        return float(loss), R.T @ grad_rot

    w_init = np.array([0.6, -2.1])
    steps = 90

    # 1. SGD (Noir)
    w_sgd = w_init.copy()
    traj_sgd = [w_sgd.copy()]
    for _ in range(steps):
        _, g = rotated_loss_and_grad(w_sgd)
        w_sgd -= 0.65 * g
        traj_sgd.append(w_sgd.copy())

    # 2. SGD + Momentum (Bleu) - grand arc / overshoot
    w_mom = w_init.copy()
    v_mom = np.zeros_like(w_mom)
    traj_mom = [w_mom.copy()]
    for _ in range(steps):
        _, g = rotated_loss_and_grad(w_mom)
        v_mom = 0.90 * v_mom + g
        w_mom -= 0.52 * v_mom
        traj_mom.append(w_mom.copy())

    # 3. RMSProp (Rouge) - route directe
    w_rms = w_init.copy()
    s_rms = np.zeros_like(w_rms)
    traj_rms = [w_rms.copy()]
    for _ in range(steps):
        _, g = rotated_loss_and_grad(w_rms)
        s_rms = 0.95 * s_rms + 0.05 * (g**2)
        w_rms -= (0.08 / (np.sqrt(s_rms) + 1e-8)) * g
        traj_rms.append(w_rms.copy())

    # 4. Adam (Violet) - équilibre parfait
    w_adam = w_init.copy()
    m_adam = np.zeros_like(w_adam)
    v_adam = np.zeros_like(w_adam)
    traj_adam = [w_adam.copy()]
    for t in range(1, steps + 1):
        _, g = rotated_loss_and_grad(w_adam)
        m_adam = 0.9 * m_adam + 0.1 * g
        v_adam = 0.999 * v_adam + 0.001 * (g**2)
        m_hat = m_adam / (1.0 - 0.9**t)
        v_hat = v_adam / (1.0 - 0.999**t)
        w_adam -= (0.11 / (np.sqrt(v_hat) + 1e-8)) * m_hat
        traj_adam.append(w_adam.copy())

    traj_sgd = np.array(traj_sgd)
    traj_mom = np.array(traj_mom)
    traj_rms = np.array(traj_rms)
    traj_adam = np.array(traj_adam)

    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    w1_grid = np.linspace(-2.5, 2.5, 350)
    w2_grid = np.linspace(-2.8, 1.8, 350)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)

    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            pt = np.array([W1[i, j], W2[i, j]])
            pt_rot = R @ pt
            Z[i, j] = 0.5 * (0.04 * (pt_rot[0]**2) + 1.1 * (pt_rot[1]**2))

    # Fond colormap continue style CS231n
    im = ax.contourf(W1, W2, Z, levels=65, cmap="turbo", alpha=0.9)
    plt.colorbar(im, ax=ax, label="Valeur de Perte $L(w_1, w_2)$")
    ax.contour(W1, W2, Z, levels=16, colors="white", alpha=0.25, linewidths=0.7)

    # Trajectoires
    ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], color="#111111", linewidth=3.2, label="SGD")
    ax.plot(traj_mom[:, 0], traj_mom[:, 1], color="#0033CC", linewidth=3.2, label="SGD+Momentum")
    ax.plot(traj_rms[:, 0], traj_rms[:, 1], color="#CC0000", linewidth=3.2, label="RMSProp")
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], color="#8A2BE2", linewidth=3.2, label="Adam")

    ax.plot(w_init[0], w_init[1], "wo", markersize=8, markeredgecolor="black", markeredgewidth=2, label="Départ")
    ax.plot(0, 0, "r*", markersize=14, label="Minimum (0, 0)")

    ax.set_title("Adam : Synthèse des Trajectoires (Réplique de la slide Stanford CS231n)", fontsize=13)
    ax.set_xlabel(r"Poids $w_1$")
    ax.set_ylabel(r"Poids $w_2$")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.8, 1.8)

    out_path = FIGURES_DIR / "03_cs231n_slide_replica.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 3 générée : {out_path.name}")


# -----------------------------------------------------------------------------
# RÉDACTION DU RAPPORT LOG 06
# -----------------------------------------------------------------------------
def write_adam_report():
    print("📝 Rédaction du rapport ADAM_REPORT.md...")

    report_content = r"""# 📓 Log 06 : Adam — Fusion de Momentum et RMSProp & Rôle de la Correction de Biais

- **Objectif** : Décortiquer l'optimiseur **Adam** (*Adaptive Moment Estimation*), comprendre la synergie entre le 1er moment (Momentum) et le 2nd moment (RMSProp), et visualiser le rôle protecteur indispensable de la **correction de biais**.
- **Référence Cours** : Stanford CS231n — Cours 3 (*Optimization & Adam*).

---

## 📊 Tableau Synthétique de la Famille des Moments

| Algorithme | Moment 1 ($m_t$ : Moyenne) | Moment 2 ($v_t$ : Variance) | Rôle & Bénéfice |
| :--- | :--- | :--- | :--- |
| **SGD + Momentum** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | Aucun | Donne de l'inertie, traverse les zones plates et annule les oscillations |
| **RMSProp** | Aucun ($m_t = g_t$) | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | Normalise chaque axe individuellement (divise par $\sqrt{v_t}$) |
| **Adam** | $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ | $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ | **Combine les deux** : vitesse d'inertie + mise à l'échelle coordonnée par coordonnée |

### Code express (Formulation complète CS231n avec Correction de Biais) :
```python
# Hyperparamètres recommandés par Stanford CS231n
beta1 = 0.9
beta2 = 0.999
learning_rate = 1e-3  # Le classique "3e-4" d'Andrej Karpathy

# Boucle d'optimisation Adam
m = beta1 * m + (1.0 - beta1) * grad                  # 1er moment (Momentum)
v = beta2 * v + (1.0 - beta2) * (grad ** 2)           # 2nd moment (RMSProp)
m_hat = m / (1.0 - beta1 ** t)                        # Correction de biais 1
v_hat = v / (1.0 - beta2 ** t)                        # Correction de biais 2
w -= (learning_rate / (np.sqrt(v_hat) + 1e-8)) * m_hat # Mise à jour
```

---

## 🎯 1. Réplique Exacte de la Slide Stanford CS231n (Les 4 Optimiseurs)

![Réplique Slide CS231n Adam](figures/03_cs231n_slide_replica.png)

> **Observation** :
> - **SGD (noir)** grimpe lentement le long de la pente et ralentit dès que le gradient faiblit.
> - **SGD+Momentum (bleu)** est emporté par son inertie (*overshoot* massif) et décrit une large boucle pendulaire.
> - **RMSProp (rouge)** ajuste immédiatement le pas selon la courbure et prend une trajectoire directe sans rebond.
> - **Adam (violet)** conjugue l'accélération d'élan de Momentum et la précision d'échelle de RMSProp pour converger de façon stable et rapide.

---

## 🧭 2. La Synergie : Décomposition des Forces

![Synergie Momentum + RMSProp = Adam](figures/01_adam_synthesis_2d.png)

> **Observation** : Momentum seul (bleu) possède une forte inertie mais est déstabilisé par la forte pente verticale ($w_2$). RMSProp seul (rouge) compense parfaitement l'asymétrie mais avance lentement le long de l'axe plat ($w_1$). **Adam (vert)** combine les deux : il amortit immédiatement l'axe raide tout en accélérant sans hésitation le long de l'axe plat vers $(0, 0)$.

---

## 🛡️ 3. Le Rôle Vital de la Correction de Biais (*Bias Correction*)

![Impact de la Correction de Biais](figures/02_bias_correction_impact.png)

> **Observation** : Initialisés à $v_0 = 0$, les premiers pas sans correction sous-estiment drastiquement la variance ($v_1 \approx 0.001 g^2$), provoquant une division par un nombre quasi-nul et un bond destructeur. La division par $(1 - \beta^t)$ normalise l'amplitude dès le pas 1 ($||\Delta w_1|| \approx \alpha$), garantissant un démarrage parfaitement stable.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 Rapport rédigé : {REPORT_FILE.name}")


def main():
    print("=" * 65)
    print("🚀 ANALYSE APPROFONDIE D'ADAM (MOMENTUM + RMSPROP)")
    print("=" * 65)
    run_synthesis_experiment()
    run_bias_correction_experiment()
    run_cs231n_slide_replica()
    write_adam_report()
    print("\n" + "=" * 65)
    print("✨ ÉTUDE D'ADAM TERMINÉE AVEC SUCCÈS !")
    print("=" * 65)


if __name__ == "__main__":
    main()

