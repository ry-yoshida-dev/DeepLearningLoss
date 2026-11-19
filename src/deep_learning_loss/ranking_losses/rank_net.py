from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..ranking_loss import RankingLoss


@dataclass(frozen=True)
class RankNetLoss(RankingLoss):
    """
    Pairwise logistic ranking loss; a simple baseline that orders every pair of items.

    RankNet loss (Burges et al., 2005): logistic loss on every pair of items whose relevance differs, so the more
    relevant item scores higher; a simple pairwise baseline that ignores where in the list a pair sits.

    The pair losses are averaged within each list; lists without a pair of different relevance are skipped.
    """

    @override
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        score_gaps: Tensor = scores.unsqueeze(2) - scores.unsqueeze(1)
        is_pair: Tensor = is_included.unsqueeze(2) & is_included.unsqueeze(1)
        is_ordered_pair: Tensor = is_pair & (relevance.unsqueeze(2) > relevance.unsqueeze(1))
        losses: Tensor = functional.softplus(-score_gaps)
        pair_counts: Tensor = is_ordered_pair.sum(dim=(1, 2))
        list_losses: Tensor = torch.where(is_ordered_pair, losses, 0.0).sum(dim=(1, 2)) / pair_counts.clamp(min=1)
        return self._average_lists(list_losses, pair_counts > 0)
