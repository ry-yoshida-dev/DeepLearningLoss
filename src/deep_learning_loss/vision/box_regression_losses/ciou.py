from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..box_regression_loss import BoxRegressionLoss


@dataclass(frozen=True)
class CIoULoss(BoxRegressionLoss):
    """
    IoU loss with center-distance and aspect-ratio penalties; the usual box loss of YOLO-style detectors.

    Complete IoU loss (Zheng et al., 2020): the DIoU loss plus ``alpha * v``, where ``v`` measures the gap between the
    aspect ratios and ``alpha = v / (1 - IoU + v)`` raises its weight as the boxes overlap; it matches overlap, center
    and shape at once.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        ious: Tensor = self._compute_ious(predictions, targets)
        aspect_penalties: Tensor = self._compute_aspect_penalties(predictions, targets)
        # The trade-off weight is a constant of each step, as in the paper.
        with torch.no_grad():
            trade_offs: Tensor = aspect_penalties / (1.0 - ious + aspect_penalties + self._EPSILON)
        center_penalties: Tensor = self._compute_center_penalties(predictions, targets)
        return 1.0 - ious + center_penalties + trade_offs * aspect_penalties
