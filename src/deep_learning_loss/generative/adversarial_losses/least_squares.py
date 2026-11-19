from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..adversarial_loss import AdversarialLoss


@dataclass(frozen=True)
class LeastSquaresAdversarialLoss(AdversarialLoss):
    """
    Least-squares GAN loss with smoother gradients; common in image-to-image translation.

    Least-squares GAN loss (Mao et al., 2017): regresses the discriminator outputs to 1 for real and 0 for generated
    samples, giving smoother gradients than the cross-entropy; common in image-to-image translation such as CycleGAN.
    """

    @override
    def compute_discriminator(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        return 0.5 * ((real_scores - 1.0).pow(2).mean() + fake_scores.pow(2).mean())

    @override
    def compute_generator(self, fake_scores: Tensor) -> Tensor:
        return 0.5 * (fake_scores - 1.0).pow(2).mean()
