from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .loss import Loss


class RankingLoss(Loss, ABC):
    """
    Loss of predicted item scores against graded relevance, for learning to rank in search and recommendation.

    Each row is one query or user with the scores of its candidate items, shape ``(lists, items)``; a higher score
    should mean a higher relevance. Only the order of the scores matters at inference. Lists of different lengths are
    padded to a common length and the padding is excluded with a mask; a list without a usable item is skipped.
    """

    def compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor | None = None) -> Tensor:
        """
        Compute the mean loss over the lists.

        Parameters
        ----------
        scores : Tensor
            Predicted scores with shape ``(lists, items)``.
        relevance : Tensor
            Non-negative graded relevance of each item, with the shape of the scores; values outside the mask may be
            anything, including NaN.
        is_included : Tensor | None
            Boolean mask of the shape of the scores marking the real items, as opposed to padding. All items when
            None.

        Returns
        -------
        Tensor
            The scalar loss; zero when no list is usable.

        Raises
        ------
        TypeError
            If is_included is not boolean.
        ValueError
            If the shapes differ or are not two-dimensional.
        """
        self._require_same_shape("scores", scores, "relevance", relevance)
        if scores.ndim != 2:
            raise ValueError(f"scores must have shape (lists, items), got {tuple(scores.shape)}")
        if is_included is None:
            is_included = torch.ones_like(scores, dtype=torch.bool)
        else:
            self._require_mask(is_included, "scores", scores)
        # Padding gets the lowest relevance and a finite score, so that it can neither win nor produce NaN.
        valid_relevance: Tensor = torch.where(is_included, relevance.to(scores.dtype), 0.0)
        valid_scores: Tensor = torch.where(is_included, scores, 0.0)
        return self._compute(valid_scores, valid_relevance, is_included)

    @staticmethod
    def _average_lists(losses: Tensor, is_valid_list: Tensor) -> Tensor:
        return torch.where(is_valid_list, losses, 0.0).sum() / is_valid_list.sum().clamp(min=1)

    @abstractmethod
    def _compute(self, scores: Tensor, relevance: Tensor, is_included: Tensor) -> Tensor:
        """
        Compute the scalar loss; scores and relevance are zero outside the mask.
        """
