# generative

## Overview

Losses and regularizers of generative models: GAN discriminators and generators, and the VAE prior.

## Components

| Component | Description |
| --------- | ----------- |
| [`adversarial_loss.py`](adversarial_loss.py) | `AdversarialLoss`: discriminator and generator losses of a GAN. |
| [adversarial_losses/](adversarial_losses/README.md) | `AdversarialLoss` implementations. |
| [`gradient_penalty.py`](gradient_penalty.py) | `GradientPenalty`: penalties on the input gradient of a GAN discriminator. |
| [gradient_penalties/](gradient_penalties/README.md) | `GradientPenalty` implementations. |
| [`latent_kl_divergence_loss.py`](latent_kl_divergence_loss.py) | `LatentKLDivergenceLoss`: divergence of a VAE posterior from the standard normal prior. |
