# 02 - Classifieurs Linéaires : SVM & Softmax

Ce module correspond au **Cours 2 de CS231n** (*Linear Classifiers*).

---

## 🎯 Notions Clés du Cours 2

### 1. Fonction de Score Linéaire
$$f(x, W, b) = W x + b$$
- $x \in \mathbb{R}^D$ : image d'entrée aplatie (ex: $32 \times 32 \times 3 = 3072$ pour CIFAR-10).
- $W \in \mathbb{R}^{C \times D}$ : matrice des poids ($C$ classes).
- $b \in \mathbb{R}^C$ : vecteur de biais.
- **Astuce du biais (*bias trick*)** : concaténer un élément $1$ au vecteur $x$ pour intégrer le biais directement dans la matrice $W$ : $f(x, W) = W x$.

### 2. Les 3 interprétations d'un classifieur linéaire
1. **Interprétation visuelle (*Template Matching*)** : Chaque ligne de la matrice $W$ correspond à un filtre ou prototype d'image appris pour une classe donnée (moyenne des apparences de la classe).
2. **Interprétation géométrique** : Chaque classe est délimitée par un hyperplan dans l'espace à $D$ dimensions. Le score correspond à la distance orientée par rapport à cet hyperplan.
3. **Interprétation algébrique** : Produit scalaire mesurant la similarité entre le vecteur d'entrée $x$ et le vecteur de poids de la classe.

### 3. Fonctions de Perte (*Loss Functions*)

#### Multi-class SVM (Hinge Loss)
La perte SVM souhaite que le score de la classe correcte dépasse tous les autres scores d'une marge $\Delta$ (généralement $\Delta = 1$) :
$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

#### Softmax (Cross-Entropy Loss)
Convertit les scores bruts en probabilités normalisées via la fonction softmax :
$$P(Y = k \mid X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$$
La perte est la log-vraisemblance négative :
$$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

### 4. Régularisation ($L_2$)
$$L = \frac{1}{N} \sum_{i=1}^N L_i + \lambda R(W) \quad \text{avec} \quad R(W) = \sum_k \sum_l W_{k, l}^2$$
- Empêche le surapprentissage (*overfitting*).
- Encourage le modèle à répartir les poids sur toutes les dimensions plutôt que de surpondérer quelques pixels isolés.

---

## 📁 Contenu du dossier

- `scripts/` : Scripts d'implémentation (SVM, Softmax, extraction des templates $W$).
- `figures/` : Visualisations exportées (templates de poids CIFAR-10, frontières linéaires 2D).
