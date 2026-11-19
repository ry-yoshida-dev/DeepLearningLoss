from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class BetaNLLLoss(DistributionLoss):
    """
    Likelihood of values strictly between 0 and 1, such as rates, proportions or probabilities.

    Beta negative log-likelihood in the mean-precision form (Ferrari and Cribari-Neto, 2004): for bounded targets such
    as click-through rates, market shares or occupancy, where a Gaussian would put mass outside the interval.

    The predictions hold ``(mean_logit, raw_concentration)`` on the last axis; the mean is ``sigmoid(mean_logit)``
    and the concentration ``softplus(raw_concentration) + minimum_concentration``, the sum of the two Beta shape
    parameters. The targets must lie strictly inside (0, 1); clamp exact zeros and ones first.

    Attributes
    ----------
    minimum_concentration : float
        Lower bound of the concentration.

    Raises
    ------
    ValueError
        If minimum_concentration is not positive.
    """

    minimum_concentration: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_concentration <= 0.0:
            raise ValueError(f"minimum_concentration must be positive, got {self.minimum_concentration}")

    @property
    @override
    def parameter_count(self) -> int:
        return 2

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        means: Tensor = torch.sigmoid(predictions[..., 0])
        concentrations: Tensor = functional.softplus(predictions[..., 1]) + self.minimum_concentration
        alphas: Tensor = means * concentrations
        betas: Tensor = concentrations - alphas
        log_likelihoods: Tensor = (
            torch.lgamma(concentrations)
            - torch.lgamma(alphas)
            - torch.lgamma(betas)
            + (alphas - 1.0) * targets.log()
            + (betas - 1.0) * torch.log1p(-targets)
        )
        return -log_likelihoods

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return torch.sigmoid(predictions[..., 0])

    @override
    def _placeholder_targets(self, targets: Tensor) -> Tensor:
        return torch.full_like(targets, 0.5)
