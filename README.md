# Stanford CS231n : Notes & Expérimentations

Journal de bord et implémentations personnelles *from scratch* suivant le cours Stanford CS231n (Deep Learning for Computer Vision).

---

## ⚡ Démarrage rapide

```bash
uv sync  # Installe l'environnement Python 3.12 (PyTorch MPS, NumPy, Matplotlib)
```

---

## 📓 Logs d'Expériences

| # | Expérience / Thème | Modèles testés | Précision Max | Rapport |
| :-: | :--- | :--- | :-: | :--- |
| **01** | **Classification d'images (Fashion-MNIST)** | kNN, Linear SVM, Softmax, Random Forest, MLP 2 couches | **86.05 %** (MLP) | [Lire le log](./01_image_classification_benchmarks/BENCHMARK_REPORT.md) |
| **02** | **Techniques de Régularisation** | Ridge, Lasso, ElasticNet, Dropout | **+4.0 %** avec Dropout | [Lire le log](./02_regularization_techniques/REGULARIZATION_REPORT.md) |
| **03** | **Optimisation 2D : GD vs Mini-Batch SGD** | Batch GD, Mini-Batch ($B=32$), Pure SGD ($B=1$) | Trajectoires 2D | [Lire le log](./03_gradient_descent_vs_sgd/OPTIMIZATION_REPORT.md) |
| **04** | **SGD + Momentum & Nesterov (NAG)** | Vanilla SGD, SGD + Momentum, Nesterov | Ravin 2D (x2.5 plus rapide) | [Lire le log](./04_momentum_and_nesterov/MOMENTUM_REPORT.md) |
| **05** | **Cas d'Échec du Gradient & Optimiseurs Avancés** | SGD, Momentum, RMSProp, Adam | Selles, Minima Locaux, Rosenbrock | [Lire le log](./05_failure_cases_and_advanced_optimizers/OPTIMIZERS_FAILURE_REPORT.md) |
| **06** | **Adam : Fusion Momentum + RMSProp** | Momentum, RMSProp, Adam (avec vs sans Bias Correction) | Trajectoires 2D & Biais | [Lire le log](./06_adam_optimizer_deep_dive/ADAM_REPORT.md) |





