from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..binary_loss import BinaryClassificationLoss


@dataclass(frozen=True)
class BinaryFocalLoss(BinaryClassificationLoss):
    """
    Binary cross-entropy that down-weights easy samples; for dense detection and class imbalance.

    Sigmoid focal loss (Lin et al., 2017): binary cross-entropy scaled by ``(1 - p_t) ** gamma``, where ``p_t`` is the
    probability of the true outcome, so the many easy negatives of dense detection weigh little.

    Attributes
    ----------
    gamma : float
        Focusing strength; 0 gives the binary cross-entropy.
    alpha : float | None
        Weight of the positive term, ``1 - alpha`` weighting the negative one; None weighs both by 1.

    Raises
    ------
    ValueError
        If gamma is negative or alpha is not in [0, 1].
    """

    gamma: float = 2.0
    alpha: float | None = 0.25

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        cross_entropies: Tensor = functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        probabilities: Tensor = torch.sigmoid(predictions)
        true_probabilities: Tensor = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        losses: Tensor = cross_entropies * (1.0 - true_probabilities).pow(self.gamma)
        if self.alpha is None:
            return losses
        return (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)) * losses
