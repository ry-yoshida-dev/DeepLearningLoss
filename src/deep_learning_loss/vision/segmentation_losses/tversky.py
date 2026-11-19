from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..segmentation_loss import SegmentationLoss


@dataclass(frozen=True)
class TverskyLoss(SegmentationLoss):
    """
    Dice variant weighting false negatives over false positives; favors recall of small structures.

    Tversky loss (Salehi et al., 2017) ``1 - TP / (TP + alpha FP + beta FN)``: a Dice loss that trades false positives
    against false negatives; a larger beta favors recall of small lesions. Alpha = beta = 0.5 gives the Dice loss.

    Attributes
    ----------
    smoothing : float
        Added to the numerator and denominator of the overlap ratio, so that a class absent from both the prediction
        and the target counts as a perfect match instead of 0 / 0.
    alpha : float
        Weight of the false positives.
    beta : float
        Weight of the false negatives.

    Raises
    ------
    ValueError
        If smoothing is negative, or alpha or beta is negative or both are zero.
    """

    smoothing: float = 1.0
    alpha: float = 0.3
    beta: float = 0.7

    def __post_init__(self) -> None:
        self._require_valid_smoothing(self.smoothing)
        if self.alpha < 0.0 or self.beta < 0.0 or self.alpha + self.beta == 0.0:
            raise ValueError(f"alpha and beta must be non-negative and not both zero, got {self.alpha} and {self.beta}")

    @override
    def _compute(self, probabilities: Tensor, truths: Tensor) -> Tensor:
        true_positives, false_positives, false_negatives = self._count_overlaps(probabilities, truths)
        overlaps: Tensor = true_positives + self.smoothing
        return (1.0 - overlaps / (overlaps + self.alpha * false_positives + self.beta * false_negatives)).mean()
