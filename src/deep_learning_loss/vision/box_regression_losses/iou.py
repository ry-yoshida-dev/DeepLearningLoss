from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..box_regression_loss import BoxRegressionLoss


@dataclass(frozen=True)
class IoULoss(BoxRegressionLoss):
    """
    Plain IoU loss; scale-invariant, but without gradient for boxes that do not overlap.

    IoU loss (Yu et al., 2016) ``1 - IoU``: optimizes the overlap metric directly instead of the coordinates, but
    gives no gradient while the predicted box misses the target; prefer ``GIoULoss`` or ``CIoULoss`` from scratch.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return 1.0 - self._compute_ious(predictions, targets)
