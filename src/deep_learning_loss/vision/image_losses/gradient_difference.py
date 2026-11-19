from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..image_loss import ImageLoss


@dataclass(frozen=True)
class GradientDifferenceLoss(ImageLoss):
    """
    Penalizes differences between image gradient magnitudes; sharpens edges in frame prediction and depth estimation.

    Gradient difference loss (Mathieu et al., 2016): the gap between the absolute horizontal and vertical differences
    of neighboring pixels in the prediction and the target, raised to ``exponent``, which pushes toward sharp edges
    where a pixel loss averages them away.

    Attributes
    ----------
    exponent : float
        Power of the gradient gaps; 1 penalizes them linearly, 2 quadratically.

    Raises
    ------
    ValueError
        If exponent is not positive.
    """

    exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.exponent <= 0.0:
            raise ValueError(f"exponent must be positive, got {self.exponent}")

    @override
    def _compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        vertical_gaps: Tensor = predictions.diff(dim=2).abs() - targets.diff(dim=2).abs()
        horizontal_gaps: Tensor = predictions.diff(dim=3).abs() - targets.diff(dim=3).abs()
        return vertical_gaps.abs().pow(self.exponent).mean() + horizontal_gaps.abs().pow(self.exponent).mean()
