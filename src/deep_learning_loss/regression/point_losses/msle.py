from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MSLELoss(PointLoss):
    """
    Squared error of log values; penalizes relative errors of non-negative targets spanning magnitudes.

    Mean squared logarithmic error ``(log(1 + prediction) - log(1 + target)) ** 2``: penalizes relative rather than
    absolute errors, for non-negative targets spanning orders of magnitude such as prices or counts.

    Predictions and targets must be greater than -1; keep predictions non-negative, for example with a softplus.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (torch.log1p(predictions) - torch.log1p(targets)).pow(2)
