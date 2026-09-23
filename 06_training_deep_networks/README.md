# 06 - Entraînement Efficace des Réseaux Profonds

## 🎯 Notions Clés Abordées

- **Initialisation des poids** : pourquoi des zéros échouent, distribution gaussienne naïve, initialisation Xavier/Glorot (pour Sigmoïde/Tanh), initialisation He/Kaiming (pour ReLU).
- **Normalisation** :
  - Batch Normalization (BatchNorm) à l'entraînement et à l'inférence.
  - Layer Normalization, Instance Normalization, Group Normalization.
- **Régularisation avancée** : Dropout, Data Augmentation (recadrages, retournements, jitter de couleur), régularisation stochastique.
- **Diagnostics d'entraînement** : ratio de mise à jour des poids, surveillance des courbes de perte (train vs val), surapprentissage volontaire sur un mini-batch (*sanity check*).
