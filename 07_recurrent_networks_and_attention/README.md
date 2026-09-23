# 07 - Réseaux Récurrents & Attention

## 🎯 Notions Clés Abordées

- **Traitement séquentiel** : un-vers-plusieurs (légendage d'images / *image captioning*), plusieurs-vers-un (analyse de sentiment), plusieurs-vers-plusieurs (traduction, prédiction vidéo).
- **Architecture RNN vanille** : état caché $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$, problème de disparition et explosion du gradient (*vanishing / exploding gradients*).
- **LSTM (Long Short-Term Memory) & GRU** : portes d'oubli, d'entrée et de sortie, cellule mémoire $c_t$.
- **Mécanismes d'Attention & Vision Transformers (ViT)** :
  - Attention scalaire et Attention produit scalaire (Query, Key, Value).
  - Découpage d'images en patchs et encodage positionnel.
