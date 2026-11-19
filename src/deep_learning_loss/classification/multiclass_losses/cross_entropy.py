from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..multiclass_loss import ClassificationLoss


@dataclass(frozen=True)
class CrossEntropyLoss(ClassificationLoss):
    """
    Default multiclass classification loss, with optional label smoothing.

    Cross-entropy of the softmax probabilities: the default for multiclass classification, optionally against
    smoothed targets to curb overconfidence.

    Attributes
    ----------
    label_smoothing : float
        Probability mass spread uniformly over the classes in the targets.

    Raises
    ------
    ValueError
        If label_smoothing is not in [0, 1).
    """

    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")

    @override
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        return functional.cross_entropy(logits, indices, reduction="none", label_smoothing=self.label_smoothing)
