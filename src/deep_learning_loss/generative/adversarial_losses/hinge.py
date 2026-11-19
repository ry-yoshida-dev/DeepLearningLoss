from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..adversarial_loss import AdversarialLoss


@dataclass(frozen=True)
class HingeAdversarialLoss(AdversarialLoss):
    """
    Hinge GAN loss; the usual pairing with spectral normalization.

    Hinge GAN loss (Lim and Ye, 2017): the discriminator pushes real scores above 1 and generated ones below -1, and
    the generator raises its scores; the usual pairing with spectral normalization, as in BigGAN.
    """

    @override
    def compute_discriminator(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        return functional.relu(1.0 - real_scores).mean() + functional.relu(1.0 + fake_scores).mean()

    @override
    def compute_generator(self, fake_scores: Tensor) -> Tensor:
        return -fake_scores.mean()
