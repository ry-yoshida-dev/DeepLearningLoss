from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..ordinal_loss import OrdinalRegressionLoss


@dataclass(frozen=True)
class CORNLoss(OrdinalRegressionLoss):
    """
    Conditional ordinal loss; consistent ranks with any output layer, often more accurate than CORAL.

    CORN loss (Shi et al., 2023): logit ``k`` predicts whether the rank exceeds ``k`` given that it exceeds ``k - 1``,
    trained only on the samples meeting that condition; the exceedance probabilities are the cumulative products of
    the conditional ones, so the predicted ranks are consistent without sharing weights between the thresholds.
    """

    @override
    def _compute(self, logits: Tensor, is_above: Tensor) -> Tensor:
        # Logit k sees the samples above threshold k - 1; every sample conditions the first logit.
        is_conditioned: Tensor = torch.cat([torch.ones_like(is_above[:, :1]), is_above[:, :-1]], dim=1)
        losses: Tensor = functional.binary_cross_entropy_with_logits(
            logits, is_above.to(logits.dtype), reduction="none"
        )
        return torch.where(is_conditioned, losses, 0.0).sum() / is_conditioned.sum().clamp(min=1)

    @override
    def _compute_exceedance_probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits).cumprod(dim=1)
