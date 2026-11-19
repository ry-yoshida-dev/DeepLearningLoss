from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..labeled_loss import LabeledEmbeddingLoss


@dataclass(frozen=True)
class TripletLoss(LabeledEmbeddingLoss):
    """
    Batch-hard triplet loss; for retrieval and re-identification.

    Batch-hard triplet loss (Hermans et al., 2017): for each anchor, its farthest same-label embedding should be closer
    by a margin than its nearest different-label one; for retrieval and re-identification.

    Anchors without both a same-label and a different-label embedding in the batch are skipped.

    Attributes
    ----------
    margin : float
        Euclidean distance by which the positive should be closer than the negative.

    Raises
    ------
    ValueError
        If margin is not positive.
    """

    margin: float = 0.2

    def __post_init__(self) -> None:
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")

    @override
    def _compute(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        distances: Tensor = self._compute_distances(embeddings)
        is_same_label: Tensor = self._compute_same_label_mask(labels)
        is_positive: Tensor = is_same_label & self._compute_off_diagonal_mask(labels)
        is_negative: Tensor = ~is_same_label
        is_valid_anchor: Tensor = is_positive.any(dim=1) & is_negative.any(dim=1)
        hardest_positives: Tensor = torch.where(is_positive, distances, float("-inf")).amax(dim=1)
        hardest_negatives: Tensor = torch.where(is_negative, distances, float("inf")).amin(dim=1)
        losses: Tensor = functional.relu(hardest_positives - hardest_negatives + self.margin)
        return torch.where(is_valid_anchor, losses, 0.0).sum() / is_valid_anchor.sum().clamp(min=1)
