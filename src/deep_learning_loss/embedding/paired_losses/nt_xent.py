from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class NTXentLoss(PairedEmbeddingLoss):
    """
    SimCLR contrastive loss; self-supervised learning from two augmentations of each sample.

    Normalized temperature-scaled cross-entropy (SimCLR; Chen et al., 2020): InfoNCE over both views pooled, so each
    view is contrasted with all other ``2 * batch - 2`` views; for self-supervised learning from augmentations.

    Attributes
    ----------
    temperature : float
        Divides the cosine similarities; smaller values concentrate on the hardest negatives.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 0.5

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        batch_size: int = first.shape[0]
        views: Tensor = functional.normalize(torch.cat([first, second]), dim=1)
        logits: Tensor = views @ views.T / self.temperature
        is_self: Tensor = torch.eye(2 * batch_size, dtype=torch.bool, device=first.device)
        logits = logits.masked_fill(is_self, float("-inf"))
        positions: Tensor = torch.arange(batch_size, device=first.device)
        partners: Tensor = torch.cat([positions + batch_size, positions])
        return functional.cross_entropy(logits, partners)
