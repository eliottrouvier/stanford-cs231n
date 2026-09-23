# Stanford CS231n : Deep Learning for Computer Vision

Dépôt personnel pour suivre le cours de Stanford **CS231n : Deep Learning for Computer Vision**.

Contient mes implémentations *from scratch*, expérimentations et visualisations personnelles codées au fil des cours.

---

## 🛠️ Environnement de développement

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour gérer l'environnement virtuel et les dépendances (Python 3.12, PyTorch avec support Apple Silicon MPS, NumPy, Matplotlib).

```bash
# Cloner le dépôt
git clone https://github.com/eliottrouvier/stanford-cs231n.git
cd stanford-cs231n

# Synchroniser l'environnement virtuel
uv sync
```

---

## 🔬 Expérimentations & Benchmarks

- 📊 **[Module 01 : Benchmark de Classification d'Images sur Fashion-MNIST](./01_image_classification_benchmarks/BENCHMARK_REPORT.md)**
  - Comparaison complète de 5 modèles : **kNN** ($k=5$), **Linear SVM** (Hinge Loss *from scratch*), **Softmax Classifier** (*from scratch*), **Random Forest** et **MLP 2 couches** (PyTorch).
  - Visualisation des templates de poids $W$, graphiques de compromis précision/vitesse et galerie d'outliers / cas de confusion.
  - Commande pour ré-exécuter le benchmark :
    ```bash
    ./.venv/bin/python 01_image_classification_benchmarks/scripts/train_benchmark.py
    ```

