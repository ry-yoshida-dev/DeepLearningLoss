import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class GaussianCRPSLoss(DistributionLoss):
    """
    Continuous ranked probability score of a Gaussian; regression with uncertainty, robust where the likelihood is not.

    Closed-form CRPS of a Gaussian (Gneiting et al., 2005): the integrated squared gap between the predicted and the
    empirical distribution function, in the units of the targets. Unlike the likelihood it stays bounded for outliers
    and does not reward an overconfident scale, which makes it the usual training and evaluation score of
    probabilistic weather and demand forecasts.

    The predictions hold ``(mean, raw_scale)`` on the last axis, as for ``GaussianNLLLoss``; the standard deviation is
    ``softplus(raw_scale) + minimum_scale``.

    Attributes
    ----------
    minimum_scale : float
        Lower bound of the standard deviation.

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
        cumulative: Tensor = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        density: Tensor = torch.exp(-0.5 * standardized.pow(2)) / math.sqrt(2.0 * math.pi)
        return scales * (standardized * (2.0 * cumulative - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0]
