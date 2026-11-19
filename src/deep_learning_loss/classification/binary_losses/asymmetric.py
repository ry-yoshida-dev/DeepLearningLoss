from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..binary_loss import BinaryClassificationLoss


@dataclass(frozen=True)
class AsymmetricLoss(BinaryClassificationLoss):
    """
    Multi-label loss that discards easy negatives; for samples with few positive labels among many.

    Asymmetric loss (Ridnik et al., 2021): focuses positives and negatives separately and discards very easy
    negatives, for multi-label classification where each sample has few positive labels among many.

    Attributes
    ----------
    gamma_positive : float
        Focusing strength of the positive term.
    gamma_negative : float
        Focusing strength of the negative term, usually larger than gamma_positive.
    clip : float
        Probability shift of the negatives; negatives predicted below it add no loss. 0 disables it.

    Raises
    ------
    ValueError
        If a gamma is negative or clip is not in [0, 1).
    """

    gamma_positive: float = 0.0
    gamma_negative: float = 4.0
    clip: float = 0.05

    def __post_init__(self) -> None:
        if self.gamma_positive < 0.0 or self.gamma_negative < 0.0:
            raise ValueError(f"gammas must be non-negative, got {self.gamma_positive} and {self.gamma_negative}")
        if not 0.0 <= self.clip < 1.0:
            raise ValueError(f"clip must be in [0, 1), got {self.clip}")

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        probabilities: Tensor = torch.sigmoid(predictions)
        shifted_probabilities: Tensor = (probabilities - self.clip).clamp(min=0.0)
        negative_log_likelihoods: Tensor = (
            torch.log((1.0 - probabilities + self.clip).clamp(max=1.0))
            if self.clip > 0.0
            else functional.logsigmoid(-predictions)
        )
        positive_terms: Tensor = (
            targets * (1.0 - probabilities).pow(self.gamma_positive) * functional.logsigmoid(predictions)
        )
        negative_terms: Tensor = (
            (1.0 - targets) * shifted_probabilities.pow(self.gamma_negative) * negative_log_likelihoods
        )
        return -(positive_terms + negative_terms)
