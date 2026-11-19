from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..distribution_loss import DistributionLoss
from .negative_binomial import NegativeBinomialNLLLoss
from .zero_inflated_poisson import ZeroInflatedPoissonNLLLoss


@dataclass(frozen=True)
class ZeroInflatedNegativeBinomialNLLLoss(DistributionLoss):
    """
    Likelihood of overdispersed counts with excess zeros, such as single-cell gene expression or intermittent demand.

    Zero-inflated negative binomial negative log-likelihood (as in scVI): a structural zero with probability ``pi``,
    otherwise a negative binomial count, for counts both overdispersed and with more zeros than that allows.

    The predictions hold ``(log_mean, raw_dispersion, zero_logit)`` on the last axis, the first two as for
    ``NegativeBinomialNLLLoss``; ``pi = sigmoid(zero_logit)``.

    Attributes
    ----------
    minimum_dispersion : float
        Lower bound of the dispersion of the negative binomial.

    Raises
    ------
    ValueError
        If minimum_dispersion is not positive.
    """

    minimum_dispersion: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_dispersion <= 0.0:
            raise ValueError(f"minimum_dispersion must be positive, got {self.minimum_dispersion}")

    @property
    @override
    def parameter_count(self) -> int:
        return 3

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        counts: NegativeBinomialNLLLoss = NegativeBinomialNLLLoss(minimum_dispersion=self.minimum_dispersion)
        count_log_likelihoods: Tensor = -counts.compute_elementwise(predictions[..., :2], targets)
        zero_log_likelihoods: Tensor = -counts.compute_elementwise(predictions[..., :2], torch.zeros_like(targets))
        return -ZeroInflatedPoissonNLLLoss.mix_with_zeros(
            count_log_likelihoods, zero_log_likelihoods, predictions[..., 2], targets
        )

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return torch.sigmoid(-predictions[..., 2]) * predictions[..., 0].exp()
