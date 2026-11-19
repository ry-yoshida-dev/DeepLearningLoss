from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class VICRegLoss(PairedEmbeddingLoss):
    """
    Variance-invariance-covariance loss; self-supervised learning without negatives or large batches.

    Variance-invariance-covariance regularization (Bardes et al., 2022): pulls the views together while keeping each
    feature's standard deviation above 1 and the features decorrelated, which prevents collapse without negatives.

    Attributes
    ----------
    invariance_weight : float
        Weight of the mean squared distance between the views.
    variance_weight : float
        Weight of the hinge on each feature's standard deviation.
    covariance_weight : float
        Weight of the squared off-diagonal covariances.

    Raises
    ------
    ValueError
        If a weight is negative.
    """

    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0

    def __post_init__(self) -> None:
        if min(self.invariance_weight, self.variance_weight, self.covariance_weight) < 0.0:
            raise ValueError(
                "weights must be non-negative, got "
                + f"{self.invariance_weight}, {self.variance_weight} and {self.covariance_weight}"
            )

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        invariance: Tensor = functional.mse_loss(first, second)
        variance: Tensor = self._compute_variance_penalty(first) + self._compute_variance_penalty(second)
        covariance: Tensor = self._compute_covariance_penalty(first) + self._compute_covariance_penalty(second)
        return (
            self.invariance_weight * invariance + self.variance_weight * variance + self.covariance_weight * covariance
        )

    @staticmethod
    def _compute_variance_penalty(embeddings: Tensor) -> Tensor:
        standard_deviations: Tensor = (embeddings.var(dim=0) + 1e-4).sqrt()
        return functional.relu(1.0 - standard_deviations).mean()

    @staticmethod
    def _compute_covariance_penalty(embeddings: Tensor) -> Tensor:
        batch_size, feature_count = embeddings.shape
        centered: Tensor = embeddings - embeddings.mean(dim=0)
        covariances: Tensor = centered.T @ centered / (batch_size - 1)
        return (covariances.pow(2).sum() - torch.diagonal(covariances).pow(2).sum()) / feature_count
