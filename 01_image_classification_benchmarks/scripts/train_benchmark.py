"""Script principal d'évaluation comparative (benchmark) sur Fashion-MNIST.

Entraîne et compare 5 familles de classifieurs :
1. k-Nearest Neighbors (kNN from scratch)
2. Linear SVM (Multiclass Hinge Loss from scratch)
3. Softmax Classifier (Cross-Entropy from scratch)
4. Forêt Aléatoire (Random Forest via scikit-learn)
5. Multi-Layer Perceptron (MLP 2 couches en PyTorch)

Génère automatiquement les métriques, graphiques d'analyse et la galerie d'outliers.
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_fashion_mnist
from models_scratch import KNearestNeighbor, LinearSVMScratch, SoftmaxClassifierScratch

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = BASE_DIR / "BENCHMARK_REPORT.md"


def set_custom_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
    })


def plot_dataset_overview(data: dict):
    """Génère un aperçu visuel des 10 classes de Fashion-MNIST."""
    set_custom_style()
    classes = data["classes"]
    X_img = data["X_train_img"]
    y = data["y_train"]

    fig, axes = plt.subplots(2, 10, figsize=(15, 3.8))
    for c_idx, c_name in enumerate(classes):
        idxs = np.where(y == c_idx)[0][:2]
        for row in range(2):
            ax = axes[row, c_idx]
            ax.imshow(X_img[idxs[row]], cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(c_name, fontsize=10, weight="bold")

    fig.suptitle("Aperçu des 10 classes de Fashion-MNIST (28x28 pixels)", y=1.05)
    out_path = FIGURES_DIR / "01_dataset_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure générée : {out_path.name}")


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Entraîne un réseau de neurones à 2 couches (MLP) en PyTorch avec support GPU MPS/CPU."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y), batch_size=batch_size, shuffle=True
    )

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    # Inférence sur le jeu de test
    model.eval()
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(test_tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    test_time = time.time() - t0

    acc = float(np.mean(preds == y_test) * 100)
    return acc, train_time, test_time, preds, probs


def plot_weight_templates(W: np.ndarray, classes: list[str], title: str, filename: str):
    """Affiche les templates visuels des poids W réarrangés en 28x28."""
    set_custom_style()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    axes = axes.flatten()

    for i in range(10):
        w_img = W[i].reshape(28, 28)
        # Normalisation locale min/max pour maximiser le contraste
        w_norm = (w_img - w_img.min()) / (w_img.max() - w_img.min() + 1e-8)
        im = axes[i].imshow(w_norm, cmap="magma")
        axes[i].set_title(classes[i], fontsize=11, weight="bold")
        axes[i].axis("off")

    fig.suptitle(title, y=0.98, fontsize=14)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure générée : {out_path.name}")


def plot_benchmark_charts(results: list[dict]):
    """Génère le graphique comparatif des performances (Accuracy et temps)."""
    set_custom_style()
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    train_times = [r["train_time"] for r in results]
    test_times = [r["test_time"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Barplot Précision
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    bars = axes[0].bar(names, accuracies, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Précision Test (%)")
    axes[0].set_title("Précision des modèles sur Fashion-MNIST")
    axes[0].tick_params(axis="x", rotation=25)

    for bar in bars:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Barplot Temps (Échelle Logarithmique)
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width / 2, train_times, width, label="Temps d'entraînement (s)", color="#4C72B0", alpha=0.85)
    axes[1].bar(x + width / 2, test_times, width, label="Temps de test / inférence (s)", color="#E15759", alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Temps en secondes (échelle log)")
    axes[1].set_title("Comparaison de la complexité temporelle (Train vs Test)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25)
    axes[1].legend()

    out_path = FIGURES_DIR / "02_benchmark_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure générée : {out_path.name}")


def plot_outliers_and_errors(
    X_test_img: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
):
    """Construit une galerie de 12 erreurs représentatives et cas limites (outliers)."""
    set_custom_style()
    misclassified_idxs = np.where(y_pred != y_test)[0]

    # Sélectionner 12 erreurs avec les plus grandes confusions
    selected_idxs = []
    # Chercher des erreurs diversifiées selon les paires de classes
    seen_pairs = set()
    for idx in misclassified_idxs:
        pair = (y_test[idx], y_pred[idx])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_idxs.append(idx)
        if len(selected_idxs) == 12:
            break

    # Si pas assez de paires distinctes, compléter
    if len(selected_idxs) < 12:
        for idx in misclassified_idxs:
            if idx not in selected_idxs:
                selected_idxs.append(idx)
            if len(selected_idxs) == 12:
                break

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    axes = axes.flatten()

    for i, idx in enumerate(selected_idxs):
        true_name = classes[y_test[idx]]
        pred_name = classes[y_pred[idx]]
        conf = probs[idx, y_pred[idx]] * 100

        axes[i].imshow(X_test_img[idx], cmap="gray")
        axes[i].axis("off")
        title_str = f"Vrai : {true_name}\nPrédit : {pred_name} ({conf:.0f}%)"
        axes[i].set_title(title_str, fontsize=10, color="#B22222", weight="bold")

    fig.suptitle(
        "Galerie d'erreurs et cas limites (Outliers / Confusions fréquentes)\nModèle Softmax",
        fontsize=14,
        y=0.98,
    )
    out_path = FIGURES_DIR / "04_error_outliers_gallery.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure générée : {out_path.name}")


def write_benchmark_report(results: list[dict], classes: list[str]):
    """Rédige automatiquement le rapport comparatif complet en Markdown."""
    table_rows = [
        "| Modèle | Type d'Approche | Précision Test | Temps Train | Temps Test (Inférence) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    for r in results:
        table_rows.append(
            f"| **{r['name']}** | {r['type']} | **{r['accuracy']:.2f} %** | {r['train_time']:.2f} s | {r['test_time']:.4f} s |"
        )
    table_content = "\n".join(table_rows)

    report_md = f"""# 📓 Log 01 : Benchmark de Classification d'Images

- **Objectif** : Comparer 5 familles de classifieurs (kNN, Linear SVM, Softmax, Random Forest, MLP 2 couches) sur des images $28 \\times 28$ pour mesurer précision et latence train/test.
- **Données** : Fashion-MNIST (10 000 train, 2 000 test, 10 classes).

![Aperçu des classes Fashion-MNIST](figures/01_dataset_overview.png)

---

## 📊 Résultats Chiffrés

{table_content}

---

## ⏱️ Temps d'Entraînement vs Inférence

![Comparaison des performances](figures/02_benchmark_comparison.png)

> **Observation** : kNN n'a aucun coût d'entraînement ($\\mathcal{{O}}(1)$) mais est très lent à l'inférence ($\\mathcal{{O}}(N \\cdot D)$) car il compare chaque pixel à toute la base. Les modèles paramétriques (SVM, Softmax, MLP) nécessitent un entraînement plus long mais prédisent quasi-instantanément ($\\mathcal{{O}}(D)$).

---

## 👁️ Templates Visuels des Poids $W$

![Templates visuels des poids W](figures/03_linear_weight_templates.png)

> **Observation** : Chaque ligne de $W$ apprend un prototype moyen par classe (formes nettes pour pantalon et bottine). Le modèle linéaire échoue face aux variations d'angles ou de styles car il ne peut mémoriser qu'un unique gabarit moyen par classe.

---

## 🚨 Galerie d'Erreurs & Outliers

![Galerie d'erreurs et confusions](figures/04_error_outliers_gallery.png)

> **Observation** : Confusions majeures entre vêtements à silhouette identique (Pull vs Manteau vs Chemise) et chaussures (Sandale vs Basket). Les détails discriminants (col, boutons, lanières) sont noyés dans la basse résolution $28 \\times 28$.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"📄 Rapport rédigé : {REPORT_FILE.name}")


def main():
    print("=" * 60)
    print("🚀 LANCEMENT DU BENCHMARK D'IMAGE CLASSIFICATION CS231n")
    print("=" * 60)

    # 1. Chargement des données
    print("\n📦 1. Chargement de Fashion-MNIST...")
    data = load_fashion_mnist(num_train=10000, num_test=2000)
    classes = data["classes"]
    X_train_flat, y_train = data["X_train_flat"], data["y_train"]
    X_test_flat, y_test = data["X_test_flat"], data["y_test"]
    X_test_img = data["X_test_img"]

    # 2. Aperçu du jeu de données
    print("🎨 Génération de l'aperçu du jeu de données...")
    plot_dataset_overview(data)

    results = []

    # 3. Modèle 1 : k-Nearest Neighbors (k=5)
    print("\n🧠 2. Entraînement et test de kNN (from scratch, k=5)...")
    knn = KNearestNeighbor(k=5)
    t0 = time.time()
    knn.fit(X_train_flat, y_train)
    knn_train_time = time.time() - t0

    t0 = time.time()
    knn_preds = knn.predict(X_test_flat)
    knn_test_time = time.time() - t0
    knn_acc = float(np.mean(knn_preds == y_test) * 100)
    print(f"   kNN Accuracy: {knn_acc:.2f}% | Train: {knn_train_time:.3f}s | Test: {knn_test_time:.2f}s")
    results.append({
        "name": "k-Nearest Neighbors",
        "type": "Instance-based (From Scratch)",
        "accuracy": knn_acc,
        "train_time": knn_train_time,
        "test_time": knn_test_time,
    })

    # 4. Modèle 2 : Linear SVM (Hinge Loss)
    print("\n⚡ 3. Entraînement de Linear SVM (from scratch)...")
    svm = LinearSVMScratch(reg=1e-3, delta=1.0)
    t0 = time.time()
    svm.fit(X_train_flat, y_train, lr=2e-2, epochs=30, batch_size=128, verbose=False)
    svm_train_time = time.time() - t0

    t0 = time.time()
    svm_preds = svm.predict(X_test_flat)
    svm_test_time = time.time() - t0
    svm_acc = float(np.mean(svm_preds == y_test) * 100)
    print(f"   Linear SVM Accuracy: {svm_acc:.2f}% | Train: {svm_train_time:.2f}s | Test: {svm_test_time:.4f}s")
    results.append({
        "name": "Linear SVM",
        "type": "Classifieur Linéaire (From Scratch)",
        "accuracy": svm_acc,
        "train_time": svm_train_time,
        "test_time": svm_test_time,
    })

    # 5. Modèle 3 : Softmax Classifier (Cross-Entropy)
    print("\n🎯 4. Entraînement de Softmax Classifier (from scratch)...")
    softmax = SoftmaxClassifierScratch(reg=1e-3)
    t0 = time.time()
    softmax.fit(X_train_flat, y_train, lr=5e-2, momentum=0.9, epochs=30, batch_size=128, verbose=False)
    softmax_train_time = time.time() - t0

    t0 = time.time()
    softmax_preds = softmax.predict(X_test_flat)
    softmax_probs = softmax.predict_proba(X_test_flat)
    softmax_test_time = time.time() - t0
    softmax_acc = float(np.mean(softmax_preds == y_test) * 100)
    print(f"   Softmax Accuracy: {softmax_acc:.2f}% | Train: {softmax_train_time:.2f}s | Test: {softmax_test_time:.4f}s")
    results.append({
        "name": "Softmax Classifier",
        "type": "Classifieur Linéaire (From Scratch)",
        "accuracy": softmax_acc,
        "train_time": softmax_train_time,
        "test_time": softmax_test_time,
    })

    # Sauvegarder les templates visuels de Softmax
    plot_weight_templates(
        softmax.W,
        classes,
        title="Templates visuels des poids W (Softmax Classifier)",
        filename="03_linear_weight_templates.png",
    )

    # 6. Modèle 4 : Forêt Aléatoire (Random Forest)
    print("\n🌲 5. Entraînement de Random Forest (scikit-learn)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train_flat, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_preds = rf.predict(X_test_flat)
    rf_test_time = time.time() - t0
    rf_acc = float(np.mean(rf_preds == y_test) * 100)
    print(f"   Random Forest Accuracy: {rf_acc:.2f}% | Train: {rf_train_time:.2f}s | Test: {rf_test_time:.4f}s")
    results.append({
        "name": "Random Forest",
        "type": "Ensemble d'arbres (scikit-learn)",
        "accuracy": rf_acc,
        "train_time": rf_train_time,
        "test_time": rf_test_time,
    })

    # 7. Modèle 5 : MLP 2 couches en PyTorch
    print("\n🔥 6. Entraînement du MLP 2 couches (PyTorch)...")
    mlp_acc, mlp_train_time, mlp_test_time, mlp_preds, mlp_probs = train_pytorch_mlp(
        X_train_flat, y_train, X_test_flat, y_test, epochs=20, lr=1e-3
    )
    print(f"   PyTorch MLP Accuracy: {mlp_acc:.2f}% | Train: {mlp_train_time:.2f}s | Test: {mlp_test_time:.4f}s")
    results.append({
        "name": "MLP (2 couches)",
        "type": "Réseau de neurones (PyTorch)",
        "accuracy": mlp_acc,
        "train_time": mlp_train_time,
        "test_time": mlp_test_time,
    })

    # 8. Visualisations comparatives
    print("\n📊 7. Tracé des graphiques comparatifs...")
    plot_benchmark_charts(results)

    # 9. Galerie des erreurs et cas limites (Outliers)
    print("🚨 8. Construction de la galerie d'outliers et d'erreurs...")
    plot_outliers_and_errors(X_test_img, y_test, softmax_preds, softmax_probs, classes)

    # 10. Rédaction du rapport Markdown
    print("📝 9. Rédaction du rapport d'expérimentation final...")
    write_benchmark_report(results, classes)

    print("\n" + "=" * 60)
    print("✨ BENCHMARK TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)


if __name__ == "__main__":
    main()
