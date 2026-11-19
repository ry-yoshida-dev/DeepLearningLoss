from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class SMAPELoss(PointLoss):
    """
    Symmetric percentage error in [0, 2]; a bounded, scale-free error for forecasting.

    Symmetric mean absolute percentage error ``2 |prediction - target| / (|prediction| + |target|)``, in [0, 2]: a
    bounded, scale-free error common in forecasting competitions.

    Attributes
    ----------
    epsilon : float
        Lower bound of the denominator, which keeps a zero prediction of a zero target from dividing by zero.

    Raises
    ------
    ValueError
        If epsilon is not positive.
    """

    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return 2.0 * (predictions - targets).abs() / (predictions.abs() + targets.abs()).clamp(min=self.epsilon)
