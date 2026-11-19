from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..labeled_loss import LabeledEmbeddingLoss


@dataclass(frozen=True)
class ContrastiveLoss(LabeledEmbeddingLoss):
    """
    Pairwise margin loss; the classic loss for Siamese verification networks.

    Pairwise contrastive loss (Hadsell et al., 2006): squared Euclidean distance for pairs of the same label and the
    squared shortfall from a margin for pairs of different labels, averaged over all pairs in the batch; the classic
    loss for Siamese networks such as signature or face verification.

    Attributes
    ----------
    margin : float
        Euclidean distance beyond which pairs of different labels add no loss.

    Raises
    ------
    ValueError
        If margin is not positive.
    """

    margin: float = 1.0

    def __post_init__(self) -> None:
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")

    @override
    def _compute(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        distances: Tensor = self._compute_distances(embeddings)
        is_same_label: Tensor = self._compute_same_label_mask(labels)
        losses: Tensor = torch.where(is_same_label, distances.pow(2), functional.relu(self.margin - distances).pow(2))
        is_pair: Tensor = torch.ones_like(is_same_label).triu(diagonal=1)
        return torch.where(is_pair, losses, 0.0).sum() / is_pair.sum().clamp(min=1)
