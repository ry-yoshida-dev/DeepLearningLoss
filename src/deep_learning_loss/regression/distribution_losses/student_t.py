import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class StudentTNLLLoss(DistributionLoss):
    """
    Heavy-tailed regression with uncertainty; a Gaussian likelihood robust to outliers.

    Student's t negative log-likelihood: a heavy-tailed Gaussian, for regression with uncertainty on data with
    outliers; the network predicts a location and a scale for each value.

    The predictions hold ``(location, raw_scale)`` on the last axis; the scale is
    ``softplus(raw_scale) + minimum_scale``.

    Attributes
    ----------
    degrees_of_freedom : float
        Tail weight, above 1 so that the mean exists; smaller values give heavier tails, large ones a Gaussian.
    minimum_scale : float
        Lower bound of the scale, which keeps the likelihood from diverging on a perfect fit.

    Raises
    ------
    ValueError
        If degrees_of_freedom is not above 1 or minimum_scale is not positive.
    """

    degrees_of_freedom: float = 3.0
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.degrees_of_freedom <= 1.0:
            raise ValueError(f"degrees_of_freedom must be above 1, got {self.degrees_of_freedom}")
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
        degrees: float = self.degrees_of_freedom
        normalizer: float = (
            math.lgamma(degrees / 2.0) - math.lgamma((degrees + 1.0) / 2.0) + 0.5 * math.log(degrees * math.pi)
        )
        return scales.log() + 0.5 * (degrees + 1.0) * torch.log1p(standardized.pow(2) / degrees) + normalizer

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0]
