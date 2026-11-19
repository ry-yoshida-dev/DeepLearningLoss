from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class PoissonNLLLoss(DistributionLoss):
    """
    Default likelihood of counts, such as demand or events, whose variance equals their mean.

    The predictions hold the logarithm of the rate on a last axis of size 1; the targets are non-negative counts.
    """

    @property
    @override
    def parameter_count(self) -> int:
        return 1

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        log_rates: Tensor = predictions[..., 0]
        return log_rates.exp() - targets * log_rates + torch.lgamma(targets + 1.0)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0].exp()
