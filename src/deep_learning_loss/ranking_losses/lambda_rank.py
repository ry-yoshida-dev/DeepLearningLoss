from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..ranking_loss import RankingLoss


@dataclass(frozen=True)
class LambdaRankLoss(RankingLoss):
    """
    Pairwise loss weighted by the NDCG change of each swap; the loss behind LambdaMART, focusing on the top.

    LambdaRank loss (Burges et al., 2006): the RankNet logistic loss of each pair whose relevance differs, weighted by
    how much swapping the two items in the current ranking would change the NDCG, so the pairs that matter at the top
    of the list dominate; it trains neural rankers the way LambdaMART trains trees.

    Lists without a relevant item are skipped.
    """

    @override
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        gains: Tensor = relevance.exp2() - 1.0
        # Current 1-based rank of each item with padding last; the weights follow the ranking but carry no gradient.
        ranking_scores: Tensor = scores.detach().masked_fill(~is_included, float("-inf"))
        ranks: Tensor = ranking_scores.argsort(dim=1, descending=True).argsort(dim=1) + 1
        discounts: Tensor = 1.0 / torch.log2(1.0 + ranks.to(scores.dtype))
        ideal_ranks: Tensor = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype)
        ideal_dcg: Tensor = (gains.sort(dim=1, descending=True).values / torch.log2(1.0 + ideal_ranks)).sum(dim=1)
        swap_changes: Tensor = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs() * (
            discounts.unsqueeze(2) - discounts.unsqueeze(1)
        ).abs()
        weights: Tensor = swap_changes / ideal_dcg.clamp(min=torch.finfo(scores.dtype).tiny).view(-1, 1, 1)
        is_pair: Tensor = is_included.unsqueeze(2) & is_included.unsqueeze(1)
        is_ordered_pair: Tensor = is_pair & (relevance.unsqueeze(2) > relevance.unsqueeze(1))
        pair_losses: Tensor = weights * functional.softplus(scores.unsqueeze(1) - scores.unsqueeze(2))
        losses: Tensor = torch.where(is_ordered_pair, pair_losses, 0.0).sum(dim=(1, 2))
        return self._average_lists(losses, ideal_dcg > 0.0)
