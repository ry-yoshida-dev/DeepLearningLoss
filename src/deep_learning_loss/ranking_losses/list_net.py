from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..ranking_loss import RankingLoss


@dataclass(frozen=True)
class ListNetLoss(RankingLoss):
    """
    Default listwise ranking loss; matches the softmax of the scores to that of the relevance.

    ListNet loss (Cao et al., 2007): cross-entropy between the softmax of the relevance and the softmax of the scores,
    a fast listwise loss that weighs the whole list at once.
    """

    @override
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        # The lowest finite value takes padding out of both softmaxes without the NaN of an infinite one.
        lowest: float = torch.finfo(scores.dtype).min
        target_probabilities: Tensor = functional.softmax(relevance.masked_fill(~is_included, lowest), dim=1)
        log_probabilities: Tensor = functional.log_softmax(scores.masked_fill(~is_included, lowest), dim=1)
        losses: Tensor = -torch.where(is_included, target_probabilities * log_probabilities, 0.0).sum(dim=1)
        return self._average_lists(losses, is_included.any(dim=1))
