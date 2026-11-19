from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..proxy_loss import ProxyEmbeddingLoss


@dataclass(frozen=True)
class CosFaceLoss(ProxyEmbeddingLoss):
    """
    Additive cosine margin softmax; simpler than ArcFace and as strong for face recognition.

    CosFace loss (Wang et al., 2018), also known as AM-Softmax: cross-entropy of the scaled cosine similarities with
    ``margin`` subtracted from the cosine to the true class proxy.

    Attributes
    ----------
    scale : float
        Multiplier of the cosine similarities, the inverse temperature of the softmax.
    margin : float
        Margin subtracted from the cosine to the true class.

    Raises
    ------
    ValueError
        If scale is not positive or margin is negative.
    """

    scale: float = 64.0
    margin: float = 0.35

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")

    @override
    def _compute(self, similarities: Tensor, labels: Tensor) -> Tensor:
        margins: Tensor = functional.one_hot(labels, similarities.shape[1]).to(similarities.dtype) * self.margin
        return functional.cross_entropy((similarities - margins) * self.scale, labels)
