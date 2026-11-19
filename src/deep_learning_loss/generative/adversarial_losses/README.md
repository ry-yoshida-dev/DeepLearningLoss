# adversarial_losses

## Overview

Discriminator and generator losses of generative adversarial networks. Each implements [`AdversarialLoss`](../adversarial_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`hinge.py`](hinge.py) | `HingeAdversarialLoss`: Hinge GAN loss; the usual pairing with spectral normalization. |
| [`least_squares.py`](least_squares.py) | `LeastSquaresAdversarialLoss`: Least-squares GAN loss with smoother gradients; common in image-to-image translation. |
| [`non_saturating.py`](non_saturating.py) | `NonSaturatingAdversarialLoss`: Default GAN loss; binary cross-entropy on the discriminator logits. |
| [`wasserstein.py`](wasserstein.py) | `WassersteinAdversarialLoss`: Wasserstein GAN loss; the critic needs a Lipschitz constraint such as a gradient penalty. |
