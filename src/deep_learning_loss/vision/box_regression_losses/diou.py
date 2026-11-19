from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..box_regression_loss import BoxRegressionLoss


@dataclass(frozen=True)
class DIoULoss(BoxRegressionLoss):
    """
    IoU loss with a center-distance penalty; converges faster than GIoU, especially for nested boxes.

    Distance IoU loss (Zheng et al., 2020) ``1 - IoU + d ** 2 / c ** 2``, with ``d`` the distance between the box
    centers and ``c`` the diagonal of the smallest box enclosing both: it moves the center directly, even when one box
    contains the other and GIoU reduces to IoU.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return 1.0 - self._compute_ious(predictions, targets) + self._compute_center_penalties(predictions, targets)
