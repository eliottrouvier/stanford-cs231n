# 🎓 Stanford CS231n : Deep Learning for Computer Vision

Dépôt personnel de progression, d'expérimentations et de visualisations interactives suivant le cours de référence de Stanford : **CS231n (Deep Learning for Computer Vision)**.

Ce projet se concentre sur l'implémentation *from scratch* des concepts fondamentaux de la vision par ordinateur et du deep learning, enrichie de visualisations visuelles haute résolution (frontières de décision, paysages de perte, templates visuels de poids $W$, filtres de convolution).

---

## 🧭 Feuille de Route & Syllabus

| Module | Thème du Cours | Statut | Dossier |
| :--- | :--- | :---: | :--- |
| **01** | **Image Classification & k-Nearest Neighbors (kNN)**<br>*(Approche data-driven, distances L1/L2, malédiction de la dimensionnalité)* | ⏳ Prêt | [`01_image_classification_knn/`](./01_image_classification_knn/) |
| **02** | **Linear Classifiers : SVM & Softmax Loss**<br>*(Interprétation géométrique/visuelle de $W$, fonction de perte Hinge & Cross-Entropy)* | 🚀 En cours | [`02_linear_classifiers/`](./02_linear_classifiers/) |
| **03** | **Optimization & Gradients**<br>*(Gradient analytique vs numérique, SGD, Momentum, RMSProp, Adam)* | 📅 Planifié | [`03_optimization_and_gradients/`](./03_optimization_and_gradients/) |
| **04** | **Neural Networks & Backpropagation**<br>*(Graphes de calcul, dérivation pas à pas, rétropropagation vectorisée)* | 📅 Planifié | [`04_neural_networks_and_backprop/`](./04_neural_networks_and_backprop/) |
| **05** | **Convolutional Neural Networks (CNNs)**<br>*(Couches Conv, Pooling, architecture LeNet/AlexNet/VGG, réceptive field)* | 📅 Planifié | [`05_convolutional_neural_networks/`](./05_convolutional_neural_networks/) |
| **06** | **Training Deep Networks**<br>*(Batch Normalization, LayerNorm, Dropout, Data Augmentation, Initialisation)* | 📅 Planifié | [`06_training_deep_networks/`](./06_training_deep_networks/) |
| **07** | **Recurrent Networks & Attention**<br>*(RNN, LSTM, mécanismes d'attention, vision transformers)* | 📅 Planifié | [`07_recurrent_networks_and_attention/`](./07_recurrent_networks_and_attention/) |
| **08** | **Generative Models**<br>*(Auto-encoders, VAEs, GANs, modèles de diffusion)* | 📅 Planifié | [`08_generative_models/`](./08_generative_models/) |

---

## 🛠️ Stack & Environnement de Développement

Ce projet utilise [**uv**](https://github.com/astral-sh/uv), le gestionnaire de paquets et d'environnements Python ultra-performant.

- **Langage** : Python 3.12
- **Calcul scientifique** : NumPy, SciPy
- **Deep Learning** : PyTorch (avec support matériel GPU Apple Silicon via `mps`)
- **Visualisations** : Matplotlib, Seaborn

### Installation rapide

1. Cloner le dépôt :
   ```bash
   git clone <URL_DU_REPO>
   cd "Stanford CS231N"
   ```

2. Installer l'environnement et les dépendances avec `uv` :
   ```bash
   uv sync
   ```

3. Exécuter un script d'expérimentation :
   ```bash
   uv run python 02_linear_classifiers/scripts/run_linear_classifier_toy.py
   ```

---

## 📂 Architecture du Projet

```text
├── README.md                      # Vue d'ensemble et tableau de bord
├── pyproject.toml                 # Dépendances gérées par uv
├── .gitignore                     # Exclusions (data/, caches, .venv/)
│
├── common/                        # Modules partagés réutilisables
│   ├── data_utils.py              # Téléchargement et chargement de CIFAR-10
│   ├── toy_data.py                # Génération de jeux de données 2D (spirales, cercles, clusters)
│   └── plot_utils.py              # Utilitaires de graphiques et sauvegarde standardisée
│
├── 01_image_classification_knn/   # Cours 1: kNN & Distances
│   ├── README.md
│   ├── scripts/
│   └── figures/
│
├── 02_linear_classifiers/         # Cours 2: Classifieurs Linéaires (SVM / Softmax)
│   ├── README.md
│   ├── scripts/
│   └── figures/
│
└── ...                            # Autres cours numérotés
```

---

## 💡 Philosophie de travail

- **Code lisible et modulaire** : chaque script peut être lancé de manière indépendante avec des arguments ajustables en ligne de commande.
- **Richesse visuelle** : toutes les exécutions exportent automatiquement des graphiques haute résolution (300 DPI) dans le sous-dossier `figures/` du module correspondant pour documenter la progression.
- **Transparence mathématique** : implémentation vectorisée des fonctions de perte et gradients analytiques en NumPy avant passage à l'autograd de PyTorch.
