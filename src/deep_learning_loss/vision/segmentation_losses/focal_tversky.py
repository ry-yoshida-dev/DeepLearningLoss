from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..segmentation_loss import SegmentationLoss


@dataclass(frozen=True)
class FocalTverskyLoss(SegmentationLoss):
    """
    Tversky loss that emphasizes poorly segmented classes; for very small, rare structures.

    Focal Tversky loss (Abraham and Khan, 2019): the Tversky loss raised to ``gamma``, which below 1 emphasizes the
    classes still segmented poorly, for highly imbalanced small structures.

    Attributes
    ----------
    smoothing : float
        Added to the numerator and denominator of the overlap ratio, so that a class absent from both the prediction
        and the target counts as a perfect match instead of 0 / 0.
    alpha : float
        Weight of the false positives.
    beta : float
        Weight of the false negatives.
    gamma : float
        Exponent of the Tversky loss; 1 gives the Tversky loss.

    Raises
    ------
    ValueError
        If smoothing is negative, alpha or beta is negative or both are zero, or gamma is not positive.
    """

    smoothing: float = 1.0
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 0.75

    def __post_init__(self) -> None:
        self._require_valid_smoothing(self.smoothing)
        if self.alpha < 0.0 or self.beta < 0.0 or self.alpha + self.beta == 0.0:
            raise ValueError(f"alpha and beta must be non-negative and not both zero, got {self.alpha} and {self.beta}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")

    @override
    def _compute(self, probabilities: Tensor, truths: Tensor) -> Tensor:
        true_positives, false_positives, false_negatives = self._count_overlaps(probabilities, truths)
        overlaps: Tensor = true_positives + self.smoothing
        indices: Tensor = overlaps / (overlaps + self.alpha * false_positives + self.beta * false_negatives)
        # The floor keeps the gradient finite for a perfect match when gamma is below 1.
        return (1.0 - indices).clamp(min=1e-7).pow(self.gamma).mean()
