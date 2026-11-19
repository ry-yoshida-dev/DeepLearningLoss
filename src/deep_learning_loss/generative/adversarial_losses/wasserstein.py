from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..adversarial_loss import AdversarialLoss


@dataclass(frozen=True)
class WassersteinAdversarialLoss(AdversarialLoss):
    """
    Wasserstein GAN loss; the critic needs a Lipschitz constraint such as a gradient penalty.

    Wasserstein GAN loss (Arjovsky et al., 2017): the critic widens the gap between its mean real and generated
    scores, estimating the Wasserstein distance; the critic must be kept 1-Lipschitz, by a gradient penalty or weight
    clipping, which this loss does not add.
    """

    @override
    def compute_discriminator(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        return fake_scores.mean() - real_scores.mean()

    @override
    def compute_generator(self, fake_scores: Tensor) -> Tensor:
        return -fake_scores.mean()
