import math
from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class LaplaceNLLLoss(DistributionLoss):
    """
    Regression with uncertainty, robust to outliers; the likelihood counterpart of the MAE.

    Laplace negative log-likelihood: the uncertainty-aware counterpart of the MAE, robust to outliers; the network
    predicts a location and a scale for each value.

    The predictions hold ``(location, raw_scale)`` on the last axis; the scale is
    ``softplus(raw_scale) + minimum_scale``.

    Attributes
    ----------
    minimum_scale : float
        Lower bound of the scale, which keeps the likelihood from diverging on a perfect fit.

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
        return scales.log() + (targets - predictions[..., 0]).abs() / scales + math.log(2.0)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0]
