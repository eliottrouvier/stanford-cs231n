# 03 - Optimisation & Gradients

## 🎯 Notions Clés Abordées

- **Le problème de l'optimisation** : paysage de perte (*loss landscape*), vallées, minima locaux, points selles.
- **Approches naïves** : recherche aléatoire (*random search*), marche aléatoire locale (*random local search*).
- **Le Gradient** : vecteur des dérivées partielles indiquant la direction de plus forte pente.
- **Gradient numérique vs analytique** :
  - Gradient numérique : calcul via différences finies $\frac{f(x+h) - f(x)}{h}$ (lent, approximatif, idéal pour le *gradient check*).
  - Gradient analytique : calcul symbolique/vectorisé exact (rapide, requiert du calcul différentiel).
- **Algorithmes d'optimisation** :
  - Descente de gradient stochastique (SGD & Mini-batch SGD)
  - SGD avec Momentum & Nesterov Momentum
  - RMSProp & AdaGrad
  - Adam (Adaptive Moment Estimation)
