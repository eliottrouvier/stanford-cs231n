# 01 - Classification d'Images & k-Nearest Neighbors (kNN)

## 🎯 Notions Clés Abordées

- **Le problème de la classification d'images** : écart sémantique (*semantic gap*), variations d'illumination, déformations, occlusions, variations intra-classe.
- **L'approche Data-Driven** : collecter un jeu de données étiqueté, entraîner un classifieur, évaluer sur un jeu de test.
- **Classifieur Nearest Neighbor (1-NN)** :
  - Distance $L_1$ (Manhattan) : $d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$
  - Distance $L_2$ (Euclidienne) : $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$
- **k-Nearest Neighbors (k-NN)** : vote majoritaire parmi les $k$ plus proches voisins pour lisser les frontières de décision.
- **Hyperparamètres & Validation** : séparation Train / Validation / Test, validation croisée (*k-fold cross-validation*).
- **Limites de kNN en vision par ordinateur** : temps de test $\mathcal{O}(N)$ prohibitif, métriques de pixels peu représentatives de la similarité perceptuelle, malédiction de la dimensionnalité (*curse of dimensionality*).

## 📁 Contenu du dossier

- `scripts/` : Scripts d'implémentation et de benchmark (calcul vectorisé sans boucle, 1 boucle, 2 boucles).
- `figures/` : Visualisations des frontières de décision 2D pour différents $k$ et métriques $L_1$/$L_2$.
