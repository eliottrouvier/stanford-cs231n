# 08 - Modèles Génératifs

## 🎯 Notions Clés Abordées

- **Taxonomie de l'apprentissage non supervisé** : estimation de densité explicite vs implicite.
- **Auto-encodeurs (AE)** : espace latent, goulot d'étranglement (*bottleneck*), perte de reconstruction.
- **Auto-encodeurs Variationnels (VAE)** :
  - Approche probabiliste : maximisation de l'ELBO (*Evidence Lower Bound*).
  - Astuce de reparamétrisation (*Reparameterization Trick*).
- **Réseaux Antagonistes Génératifs (GANs)** :
  - Jeu minimax à 2 joueurs : Générateur $G$ vs Discriminateur $D$.
  - Fonction de coût minimax et Wasserstein GAN (WGAN).
- **Modèles de Diffusion** : processus de bruitage direct (*forward*) et débruitage appris (*reverse*).
