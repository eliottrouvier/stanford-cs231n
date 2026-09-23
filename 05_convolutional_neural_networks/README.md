# 05 - Réseaux de Neurones Convolutifs (CNNs)

## 🎯 Notions Clés Abordées

- **Motivation biologique & spatiale** : invariance par translation, connectivité locale.
- **La couche de Convolution (Conv Layer)** :
  - Filtres / Noyaux (*kernels*), cartes d'activation (*feature maps*).
  - Hyperparamètres spatiaux : taille de filtre $F$, pas (*stride*) $S$, remplissage (*padding*) $P$.
  - Formule de la dimension de sortie : $W_{out} = \frac{W - F + 2P}{S} + 1$.
  - Partage de paramètres (*parameter sharing*).
- **Couches de Pooling** : Max Pooling, Average Pooling, réduction spatiale sans paramètres appris.
- **Architectures classiques** : LeNet-5, AlexNet, VGGNet, GoogLeNet (Inception), ResNet (connexions résiduelles).
