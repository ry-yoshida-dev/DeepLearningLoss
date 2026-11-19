from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..adversarial_loss import AdversarialLoss


@dataclass(frozen=True)
class NonSaturatingAdversarialLoss(AdversarialLoss):
    """
    Default GAN loss; binary cross-entropy on the discriminator logits.

    Non-saturating GAN loss (Goodfellow et al., 2014): binary cross-entropy on the discriminator logits, with the
    generator maximizing the log-probability of being judged real; the standard choice, as in StyleGAN.
    """

    @override
    def compute_discriminator(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        return functional.softplus(-real_scores).mean() + functional.softplus(fake_scores).mean()

    @override
    def compute_generator(self, fake_scores: Tensor) -> Tensor:
        return functional.softplus(-fake_scores).mean()
