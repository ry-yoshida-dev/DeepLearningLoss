import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class LogNormalNLLLoss(DistributionLoss):
    """
    Likelihood of positive values spanning orders of magnitude, such as prices, incomes or latencies.

    Log-normal negative log-likelihood: a Gaussian on the logarithm of positive targets, for heavy right tails and
    multiplicative noise.

    The predictions hold ``(log_location, raw_scale)`` on the last axis: the mean and the standard deviation
    ``softplus(raw_scale) + minimum_scale`` of the logarithm of the target. The targets must be positive.

    Attributes
    ----------
    minimum_scale : float
        Lower bound of the standard deviation of the logarithm.

    Raises
    ------
    ValueError
        If minimum_scale is not positive.
    """

    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_scale <= 0.0:
            raise ValueError(f"minimum_scale must be positive, got {self.minimum_scale}")

    @property
    @override
    def parameter_count(self) -> int:
        return 2

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        scales: Tensor = functional.softplus(predictions[..., 1]) + self.minimum_scale
        log_targets: Tensor = targets.log()
        standardized: Tensor = (log_targets - predictions[..., 0]) / scales
        return log_targets + scales.log() + 0.5 * standardized.pow(2) + 0.5 * math.log(2.0 * math.pi)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        scales: Tensor = functional.softplus(predictions[..., 1]) + self.minimum_scale
        return (predictions[..., 0] + 0.5 * scales.pow(2)).exp()

    @override
    def _placeholder_targets(self, targets: Tensor) -> Tensor:
        return torch.ones_like(targets)
