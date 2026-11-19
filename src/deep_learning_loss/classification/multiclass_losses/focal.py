from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..multiclass_loss import ClassificationLoss


@dataclass(frozen=True)
class FocalLoss(ClassificationLoss):
    """
    Cross-entropy that down-weights easy samples; for class imbalance.

    Focal loss (Lin et al., 2017): cross-entropy scaled by ``(1 - p) ** gamma``, where ``p`` is the predicted
    probability of the true class, so well-classified samples weigh less and hard ones dominate under class imbalance.

    Attributes
    ----------
    gamma : float
        Focusing strength; 0 gives the cross-entropy.

    Raises
    ------
    ValueError
        If gamma is negative.
    """

    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

    @override
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        log_probabilities: Tensor = functional.log_softmax(logits, dim=1).gather(1, indices.unsqueeze(1)).squeeze(1)
        return -((1.0 - log_probabilities.exp()).pow(self.gamma)) * log_probabilities
