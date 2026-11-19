import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..proxy_loss import ProxyEmbeddingLoss


@dataclass(frozen=True)
class ArcFaceLoss(ProxyEmbeddingLoss):
    """
    Default proxy loss; additive angular margin softmax, the standard for face recognition.

    ArcFace loss (Deng et al., 2019): cross-entropy of the scaled cosine similarities with the angle to the true
    class proxy enlarged by ``margin``, which leaves a geodesic gap between classes on the hypersphere.

    Attributes
    ----------
    scale : float
        Multiplier of the cosine similarities, the inverse temperature of the softmax.
    margin : float
        Angular margin in radians added to the angle to the true class.

    Raises
    ------
    ValueError
        If scale is not positive or margin is not in [0, pi / 2).
    """

    scale: float = 64.0
    margin: float = 0.5

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not 0.0 <= self.margin < math.pi / 2.0:
            raise ValueError(f"margin must be in [0, pi / 2), got {self.margin}")

    @override
    def _compute(self, similarities: Tensor, labels: Tensor) -> Tensor:
        true_cosines: Tensor = similarities.gather(1, labels.unsqueeze(1))
        # The floor keeps the gradient of the square root finite at cosines of exactly 1.
        true_sines: Tensor = (1.0 - true_cosines.pow(2)).clamp(min=1e-7).sqrt()
        margin_cosines: Tensor = true_cosines * math.cos(self.margin) - true_sines * math.sin(self.margin)
        # Beyond pi - margin the cosine of the enlarged angle would rise again, so a linear penalty takes over.
        is_monotonic: Tensor = true_cosines > math.cos(math.pi - self.margin)
        linear_cosines: Tensor = true_cosines - math.sin(math.pi - self.margin) * self.margin
        adjusted_cosines: Tensor = torch.where(is_monotonic, margin_cosines, linear_cosines)
        logits: Tensor = similarities.scatter(1, labels.unsqueeze(1), adjusted_cosines) * self.scale
        return functional.cross_entropy(logits, labels)
