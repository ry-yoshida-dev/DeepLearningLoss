from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class InfoNCELoss(PairedEmbeddingLoss):
    """
    Default contrastive loss for paired views; the CLIP loss for aligning two modalities.

    InfoNCE loss (van den Oord et al., 2018): cross-entropy of picking each first view's partner among all second
    views by cosine similarity; symmetric, it is the CLIP loss for aligning two modalities such as images and text.

    Attributes
    ----------
    temperature : float
        Divides the cosine similarities; smaller values concentrate on the hardest negatives.
    is_symmetric : bool
        Whether to also pick each second view's partner among the first views and average both directions.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 0.07
    is_symmetric: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        logits: Tensor = functional.normalize(first, dim=1) @ functional.normalize(second, dim=1).T / self.temperature
        partners: Tensor = torch.arange(first.shape[0], device=first.device)
        loss: Tensor = functional.cross_entropy(logits, partners)
        if not self.is_symmetric:
            return loss
        return 0.5 * (loss + functional.cross_entropy(logits.T, partners))
