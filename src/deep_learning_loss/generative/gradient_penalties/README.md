# gradient_penalties

## Overview

Penalties on the input gradient of a GAN discriminator, added to its loss. Each implements [`GradientPenalty`](../gradient_penalty.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`r1.py`](r1.py) | `R1GradientPenalty`: Zero-centered penalty on real samples; the standard regularizer of StyleGAN and modern GANs. |
| [`wasserstein.py`](wasserstein.py) | `WassersteinGradientPenalty`: One-centered penalty on interpolated samples; the WGAN-GP constraint that keeps the critic 1-Lipschitz. |
