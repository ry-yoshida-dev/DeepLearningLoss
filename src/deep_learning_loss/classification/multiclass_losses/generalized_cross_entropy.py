from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..multiclass_loss import ClassificationLoss


@dataclass(frozen=True)
class GeneralizedCrossEntropyLoss(ClassificationLoss):
    """
    Cross-entropy robust to noisy labels, between cross-entropy and MAE.

    Generalized cross-entropy (Zhang and Sabuncu, 2018) ``(1 - p ** q) / q``, where ``p`` is the predicted probability
    of the true class: robust to noisy labels, between the cross-entropy (q toward 0) and the MAE (q = 1).

    Attributes
    ----------
    q : float
        Robustness, in (0, 1]; larger values fit mislabeled samples less but train more slowly.

    Raises
    ------
    ValueError
        If q is not in (0, 1].
    """

    q: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 < self.q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {self.q}")

    @override
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        log_probabilities: Tensor = functional.log_softmax(logits, dim=1).gather(1, indices.unsqueeze(1)).squeeze(1)
        return (1.0 - (self.q * log_probabilities).exp()) / self.q
