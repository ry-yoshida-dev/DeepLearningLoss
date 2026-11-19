from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..multiclass_loss import ClassificationLoss


@dataclass(frozen=True)
class Poly1Loss(ClassificationLoss):
    """
    Cross-entropy plus a linear term; a drop-in replacement that often improves accuracy.

    Poly-1 loss (Leng et al., 2022): cross-entropy plus ``epsilon * (1 - p)``, where ``p`` is the predicted
    probability of the true class; a drop-in replacement for cross-entropy that often improves accuracy.

    Attributes
    ----------
    epsilon : float
        Weight of the added term; positive values push confident predictions further, 0 gives the cross-entropy.

    Raises
    ------
    ValueError
        If epsilon is below -1, where the loss would stop decreasing with the true-class probability.
    """

    epsilon: float = 1.0

    def __post_init__(self) -> None:
        if self.epsilon < -1.0:
            raise ValueError(f"epsilon must be at least -1, got {self.epsilon}")

    @override
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        log_probabilities: Tensor = functional.log_softmax(logits, dim=1).gather(1, indices.unsqueeze(1)).squeeze(1)
        return -log_probabilities + self.epsilon * (1.0 - log_probabilities.exp())
