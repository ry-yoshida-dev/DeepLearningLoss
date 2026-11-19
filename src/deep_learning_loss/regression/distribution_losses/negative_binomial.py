from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class NegativeBinomialNLLLoss(DistributionLoss):
    """
    Likelihood of overdispersed counts, whose variance exceeds their mean, such as intermittent demand.

    Negative binomial negative log-likelihood: for overdispersed counts, whose variance exceeds their mean, as in
    intermittent demand (DeepAR) or gene expression.

    The predictions hold ``(log_mean, raw_dispersion)`` on the last axis; the variance is
    ``mean + dispersion * mean ** 2`` with dispersion ``softplus(raw_dispersion) + minimum_dispersion``.

    Attributes
    ----------
    minimum_dispersion : float
        Lower bound of the dispersion; near it the distribution approaches the Poisson.

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
        return 2

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        log_means: Tensor = predictions[..., 0]
        dispersions: Tensor = functional.softplus(predictions[..., 1]) + self.minimum_dispersion
        total_counts: Tensor = 1.0 / dispersions
        # log(1 + dispersion * mean), shared by the log-probabilities of success and failure, without overflowing exp.
        log_ratios: Tensor = functional.softplus(log_means + dispersions.log())
        log_likelihoods: Tensor = (
            torch.lgamma(targets + total_counts)
            - torch.lgamma(total_counts)
            - torch.lgamma(targets + 1.0)
            - total_counts * log_ratios
            + targets * (log_means + dispersions.log() - log_ratios)
        )
        return -log_likelihoods

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return predictions[..., 0].exp()
