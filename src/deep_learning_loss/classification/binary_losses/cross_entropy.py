from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..binary_loss import BinaryClassificationLoss


@dataclass(frozen=True)
class BinaryCrossEntropyLoss(BinaryClassificationLoss):
    """
    Default binary and multi-label classification loss, with an optional positive weight.

    Binary cross-entropy of the sigmoid probabilities.

    Attributes
    ----------
    positive_weight : float
        Weight of the positive term; set it to about ``negatives / positives`` against class imbalance.

    Raises
    ------
    ValueError
        If positive_weight is not positive.
    """

    positive_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.positive_weight <= 0.0:
            raise ValueError(f"positive_weight must be positive, got {self.positive_weight}")

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        positive_terms: Tensor = targets * functional.logsigmoid(predictions)
        negative_terms: Tensor = (1.0 - targets) * functional.logsigmoid(-predictions)
        return -(self.positive_weight * positive_terms + negative_terms)
