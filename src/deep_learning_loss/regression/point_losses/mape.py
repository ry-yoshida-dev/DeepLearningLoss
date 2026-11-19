from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MAPELoss(PointLoss):
    """
    Absolute percentage error; a scale-free error for targets far from zero.

    Mean absolute percentage error ``|prediction - target| / |target|``, as a fraction: scale-free, for targets far
    from zero; it penalizes over-prediction more than under-prediction.

    Attributes
    ----------
    epsilon : float
        Lower bound of the denominator, which keeps targets near zero from dividing by zero.

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
        return (predictions - targets).abs() / targets.abs().clamp(min=self.epsilon)
