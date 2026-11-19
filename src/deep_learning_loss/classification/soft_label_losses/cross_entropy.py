from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..soft_label_loss import SoftLabelClassificationLoss


@dataclass(frozen=True)
class SoftLabelCrossEntropyLoss(SoftLabelClassificationLoss):
    """
    Default loss for soft targets; the cross-entropy that Mixup, CutMix and pseudo-labeling train with.

    Cross-entropy ``-sum_c p_c log q_c`` between the target probabilities and the softmax of the logits; with one-hot
    targets it equals the ordinary cross-entropy.
    """

    @override
    def _compute_elementwise(self, logits: Tensor, probabilities: Tensor) -> Tensor:
        return -(probabilities * functional.log_softmax(logits, dim=1)).sum(dim=1)
