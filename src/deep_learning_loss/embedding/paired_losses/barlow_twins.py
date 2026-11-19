from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class BarlowTwinsLoss(PairedEmbeddingLoss):
    """
    Cross-correlation loss; self-supervised learning without negatives, working with small batches.

    Barlow Twins loss (Zbontar et al., 2021): drives the cross-correlation matrix of the two views to the identity, so
    features agree across views and carry no redundancy; needs no negatives and works with small batches.

    Attributes
    ----------
    redundancy_weight : float
        Weight of the off-diagonal (redundancy) terms against the diagonal (invariance) terms.

    Raises
    ------
    ValueError
        If redundancy_weight is negative.
    """

    redundancy_weight: float = 5e-3

    def __post_init__(self) -> None:
        if self.redundancy_weight < 0.0:
            raise ValueError(f"redundancy_weight must be non-negative, got {self.redundancy_weight}")

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        batch_size: int = first.shape[0]
        first_standardized: Tensor = (first - first.mean(dim=0)) / first.std(dim=0).clamp(min=1e-6)
        second_standardized: Tensor = (second - second.mean(dim=0)) / second.std(dim=0).clamp(min=1e-6)
        correlations: Tensor = first_standardized.T @ second_standardized / batch_size
        diagonal: Tensor = torch.diagonal(correlations)
        invariance: Tensor = (diagonal - 1.0).pow(2).sum()
        redundancy: Tensor = correlations.pow(2).sum() - diagonal.pow(2).sum()
        return invariance + self.redundancy_weight * redundancy
