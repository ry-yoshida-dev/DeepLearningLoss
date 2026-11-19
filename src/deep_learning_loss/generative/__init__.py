from .adversarial_loss import AdversarialLoss
from .adversarial_losses import (
    HingeAdversarialLoss,
    LeastSquaresAdversarialLoss,
    NonSaturatingAdversarialLoss,
    WassersteinAdversarialLoss,
)
from .gradient_penalties import R1GradientPenalty, WassersteinGradientPenalty
from .gradient_penalty import GradientPenalty
from .latent_kl_divergence_loss import LatentKLDivergenceLoss

__all__ = [
    "AdversarialLoss",
    "HingeAdversarialLoss",
    "LeastSquaresAdversarialLoss",
    "NonSaturatingAdversarialLoss",
    "WassersteinAdversarialLoss",
    "GradientPenalty",
    "R1GradientPenalty",
    "WassersteinGradientPenalty",
    "LatentKLDivergenceLoss",
]
