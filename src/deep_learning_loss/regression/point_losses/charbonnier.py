from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class CharbonnierLoss(PointLoss):
    """
    Differentiable absolute error; common in image restoration such as super-resolution.

    Charbonnier loss ``sqrt(error ** 2 + epsilon ** 2)``: a differentiable absolute error, common in image restoration
    such as super-resolution and denoising.

    Attributes
    ----------
    epsilon : float
        Scale below which the loss becomes quadratic; the loss of a zero error.

    Raises
    ------
    ValueError
        If epsilon is not positive.
    """

    epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ((predictions - targets).pow(2) + self.epsilon**2).sqrt()
