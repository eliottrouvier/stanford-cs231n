# 📓 Log 01 : Benchmark de Classification d'Images

- **Objectif** : Comparer 5 familles de classifieurs (kNN, Linear SVM, Softmax, Random Forest, MLP 2 couches) sur des images $28 \times 28$ pour mesurer précision et latence train/test.
- **Données** : Fashion-MNIST (10 000 train, 2 000 test, 10 classes).

![Aperçu des classes Fashion-MNIST](figures/01_dataset_overview.png)

---

## 📊 Résultats Chiffrés

| Modèle | Type d'Approche | Précision Test | Temps Train | Temps Test (Inférence) |
| :--- | :--- | :---: | :---: | :---: |
| **k-Nearest Neighbors ($k=5$)** | Instance-based (*From Scratch*) | **81.95 %** | 0.00 s | 0.2190 s |
| **Linear SVM (Hinge Loss)** | Classifieur Linéaire (*From Scratch*) | **81.40 %** | 0.26 s | 0.0007 s |
| **Softmax Classifier** | Classifieur Linéaire (*From Scratch*) | **83.15 %** | 0.28 s | 0.0012 s |
| **Random Forest (100 arbres)** | Ensemble d'arbres (`scikit-learn`) | **84.95 %** | 0.86 s | 0.0127 s |
| **MLP (2 couches ReLU)** | Réseau de neurones (`PyTorch`) | **86.05 %** | 3.55 s | 0.1377 s |

---

## ⏱️ Temps d'Entraînement vs Inférence

![Comparaison des performances](figures/02_benchmark_comparison.png)

> **Observation** : kNN n'a aucun coût d'entraînement ($\mathcal{O}(1)$) mais est très lent à l'inférence ($\mathcal{O}(N \cdot D)$) car il compare chaque pixel à toute la base. Les modèles paramétriques (SVM, Softmax, MLP) nécessitent un entraînement plus long mais prédisent quasi-instantanément ($\mathcal{O}(D)$).

---

## 👁️ Templates Visuels des Poids $W$

![Templates visuels des poids W](figures/03_linear_weight_templates.png)

> **Observation** : Chaque ligne de $W$ apprend un prototype moyen par classe (formes nettes pour pantalon et bottine). Le modèle linéaire échoue face aux variations d'angles ou de styles car il ne peut mémoriser qu'un unique gabarit moyen par classe.

---

## 🚨 Galerie d'Erreurs & Outliers

![Galerie d'erreurs et confusions](figures/04_error_outliers_gallery.png)

> **Observation** : Confusions majeures entre vêtements à silhouette identique (Pull vs Manteau vs Chemise) et chaussures (Sandale vs Basket). Les détails discriminants (col, boutons, lanières) sont noyés dans la basse résolution $28 \times 28$.
