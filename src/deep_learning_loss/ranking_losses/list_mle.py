from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..ranking_loss import RankingLoss


@dataclass(frozen=True)
class ListMLELoss(RankingLoss):
    """
    Listwise likelihood of the full order by relevance; for when the whole order matters.

    ListMLE loss (Xia et al., 2008): negative log-likelihood of the order by relevance under the Plackett-Luce model of
    the scores, a listwise loss for when the full order matters; ties in relevance are broken arbitrarily.
    """

    @override
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        # Padding is placed last and given the lowest finite score, so that it drops out of every normalizer.
        order: Tensor = relevance.masked_fill(~is_included, -1.0).argsort(dim=1, descending=True)
        is_ordered_included: Tensor = is_included.gather(1, order)
        lowest: float = torch.finfo(scores.dtype).min
        ordered_scores: Tensor = scores.gather(1, order).masked_fill(~is_ordered_included, lowest)
        # Log-normalizer of each step: logsumexp over the items not yet placed.
        remaining_log_sums: Tensor = torch.logcumsumexp(ordered_scores.flip(1), dim=1).flip(1)
        steps: Tensor = torch.where(is_ordered_included, remaining_log_sums - ordered_scores, 0.0)
        return self._average_lists(steps.sum(dim=1), is_included.any(dim=1))
