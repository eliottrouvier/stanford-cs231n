# 04 - Réseaux de Neurones & Rétropropagation

## 🎯 Notions Clés Abordées

- **Limites des classifieurs linéaires** : impossibilité de séparer les données non linéaires (ex: problème XOR, dataset spirale).
- **Graphes de calcul & Rétropropagation (*Computational Graphs & Backprop*)** :
  - Règle de dérivation en chaîne (*Chain Rule*) : $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$
  - Passes avant (*forward pass*) et passes arrière (*backward pass*).
  - Portes élémentaires : addition (distributeur de gradient), multiplication (échangeur de gradient), max (routeur de gradient).
- **Réseau de neurones à 2 couches** :
  - $f = W_2 \cdot \max(0, W_1 x + b_1) + b_2$
  - Rôle des fonctions d'activation non linéaires (ReLU, Sigmoïde, Tanh, LeakyReLU, GeLU).
- **Théorème d'approximation universelle**.
