from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MAELoss(PointLoss):
    """
    Mean absolute error; robust to outliers, predicting the conditional median.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).abs()
