from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..soft_label_loss import SoftLabelClassificationLoss


@dataclass(frozen=True)
class SoftLabelFocalLoss(SoftLabelClassificationLoss):
    """
    Focal loss for soft targets; down-weights well-classified classes under Mixup or label noise.

    Soft-target focal loss ``-sum_c p_c (1 - q_c) ** gamma log q_c``: the focal loss (Lin et al., 2017) with each
    class term weighted by its target probability; with one-hot targets it equals the ordinary focal loss.

    Attributes
    ----------
    gamma : float
        Focusing exponent; 0 gives the soft-label cross-entropy.

    Raises
    ------
    ValueError
        If gamma is negative.
    """

    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

    @override
    def _compute_elementwise(self, logits: Tensor, probabilities: Tensor) -> Tensor:
        log_probabilities: Tensor = functional.log_softmax(logits, dim=1)
        focusing: Tensor = (1.0 - log_probabilities.exp()).pow(self.gamma)
        return -(probabilities * focusing * log_probabilities).sum(dim=1)
