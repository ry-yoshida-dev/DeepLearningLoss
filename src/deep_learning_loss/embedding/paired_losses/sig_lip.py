from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class SigLIPLoss(PairedEmbeddingLoss):
    """
    Sigmoid pairwise loss for aligning two modalities; scales to large batches better than the CLIP softmax.

    SigLIP loss (Zhai et al., 2023): an independent logistic loss on every pair of the batch, positive for the
    corresponding rows and negative for all others, so it needs no normalization over the batch and works well with
    both small and very large batches. The paper learns the scale and the bias; here they are fixed.

    Attributes
    ----------
    scale : float
        Multiplier of the cosine similarities, the inverse temperature.
    bias : float
        Added to the scaled similarities; its negative initial value offsets the many negative pairs.

    Raises
    ------
    ValueError
        If scale is not positive.
    """

    scale: float = 10.0
    bias: float = -10.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        logits: Tensor = (
            functional.normalize(first, dim=1) @ functional.normalize(second, dim=1).T * self.scale + self.bias
        )
        signs: Tensor = 2.0 * torch.eye(first.shape[0], dtype=logits.dtype, device=logits.device) - 1.0
        return -functional.logsigmoid(signs * logits).sum() / first.shape[0]
