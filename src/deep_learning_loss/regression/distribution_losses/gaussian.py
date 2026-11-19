import math
from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class GaussianNLLLoss(DistributionLoss):
    """
    Default regression with uncertainty; predicts a mean and a standard deviation for each value.

    Gaussian negative log-likelihood: the default for regression with uncertainty; the network predicts a mean and a
    standard deviation for each value.

    The predictions hold ``(mean, raw_scale)`` on the last axis; the standard deviation is
    ``softplus(raw_scale) + minimum_scale``.

    Attributes
    ----------
    minimum_scale : float
        Lower bound of the standard deviation, which keeps the likelihood from diverging on a perfect fit.

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
        standardized: Tensor = (targets - predictions[..., 0]) / scales
        return scales.log() + 0.5 * standardized.pow(2) + 0.5 * math.log(2.0 * math.pi)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0]
