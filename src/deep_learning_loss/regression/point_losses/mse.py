from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MSELoss(PointLoss):
    """
    Default regression loss; predicts the conditional mean, and large errors dominate.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).pow(2)
