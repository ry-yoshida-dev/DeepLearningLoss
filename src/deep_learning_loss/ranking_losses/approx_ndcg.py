from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..ranking_loss import RankingLoss


@dataclass(frozen=True)
class ApproxNDCGLoss(RankingLoss):
    """
    Differentiable NDCG; optimizes the ranking metric directly, weighing the top of the list most.

    ApproxNDCG loss (Qin et al., 2010): one minus the NDCG with each rank replaced by a sigmoid-smoothed rank, so it
    optimizes the metric search engines report, weighing the top of the list most.

    Lists without a relevant item are skipped.

    Attributes
    ----------
    temperature : float
        Smoothness of the approximate ranks; smaller values follow the true ranks more closely but give sparser
        gradients.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @override
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        # Rank of item i: 1 plus the smoothed count of real items scoring above it; the self term gives sigmoid(0).
        pairwise: Tensor = torch.sigmoid((scores.unsqueeze(1) - scores.unsqueeze(2)) / self.temperature)
        approximate_ranks: Tensor = 0.5 + (pairwise * is_included.unsqueeze(1)).sum(dim=2)
        gains: Tensor = torch.where(is_included, relevance.exp2() - 1.0, 0.0)
        dcg: Tensor = (gains / torch.log2(1.0 + approximate_ranks)).sum(dim=1)
        ideal_gains: Tensor = gains.sort(dim=1, descending=True).values
        ideal_ranks: Tensor = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype)
        ideal_dcg: Tensor = (ideal_gains / torch.log2(1.0 + ideal_ranks)).sum(dim=1)
        losses: Tensor = 1.0 - dcg / ideal_dcg.clamp(min=torch.finfo(scores.dtype).tiny)
        return self._average_lists(losses, ideal_dcg > 0.0)
