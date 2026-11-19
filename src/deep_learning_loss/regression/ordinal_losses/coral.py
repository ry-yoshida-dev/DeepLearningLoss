from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..ordinal_loss import OrdinalRegressionLoss


@dataclass(frozen=True)
class CORALLoss(OrdinalRegressionLoss):
    """
    Default ordinal loss; one binary task per threshold, consistent when the logits share their weights.

    CORAL loss (Cao et al., 2020): binary cross-entropy of each logit ``k`` against whether the rank exceeds ``k``,
    summed over the thresholds. The predicted ranks are consistent when the logits share one weight vector and differ
    only by their biases, as in a final linear layer with one output and ``ranks - 1`` biases.
    """

    @override
    def _compute(self, logits: Tensor, is_above: Tensor) -> Tensor:
        losses: Tensor = functional.binary_cross_entropy_with_logits(
            logits, is_above.to(logits.dtype), reduction="none"
        )
        return losses.sum(dim=1).mean()

    @override
    def _compute_exceedance_probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)
