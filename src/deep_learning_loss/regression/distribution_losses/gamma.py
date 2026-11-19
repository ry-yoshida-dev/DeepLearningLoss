from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class GammaNLLLoss(DistributionLoss):
    """
    Likelihood of positive, right-skewed values, such as durations, amounts or rainfall.

    Gamma negative log-likelihood: for positive targets whose spread grows with their size, such as waiting times,
    insurance claims or precipitation amounts.

    The predictions hold ``(log_mean, raw_shape)`` on the last axis; the shape is
    ``softplus(raw_shape) + minimum_shape`` and the variance is ``mean ** 2 / shape``. The targets must be positive.

    Attributes
    ----------
    minimum_shape : float
        Lower bound of the shape parameter.

    Raises
    ------
    ValueError
        If minimum_shape is not positive.
    """

    minimum_shape: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_shape <= 0.0:
            raise ValueError(f"minimum_shape must be positive, got {self.minimum_shape}")

    @property
    @override
    def parameter_count(self) -> int:
        return 2

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        log_means: Tensor = predictions[..., 0]
        shapes: Tensor = functional.softplus(predictions[..., 1]) + self.minimum_shape
        log_rates: Tensor = shapes.log() - log_means
        log_likelihoods: Tensor = (
            shapes * log_rates - torch.lgamma(shapes) + (shapes - 1.0) * targets.log() - log_rates.exp() * targets
        )
        return -log_likelihoods

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0].exp()

    @override
    def _placeholder_targets(self, targets: Tensor) -> Tensor:
        return torch.ones_like(targets)
