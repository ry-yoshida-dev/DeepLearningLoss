from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..segmentation_loss import SegmentationLoss


@dataclass(frozen=True)
class DiceLoss(SegmentationLoss):
    """
    Default overlap loss for segmentation; robust to class imbalance, usually added to cross-entropy.

    Soft Dice loss ``1 - 2 TP / (2 TP + FP + FN)``: the default overlap loss for segmentation, especially medical
    images with small structures; pair it with cross-entropy for stable training.

    Attributes
    ----------
    smoothing : float
        Added to the numerator and denominator of the overlap ratio, so that a class absent from both the prediction
        and the target counts as a perfect match instead of 0 / 0.

    Raises
    ------
    ValueError
        If smoothing is negative.
    """

    smoothing: float = 1.0

    def __post_init__(self) -> None:
        self._require_valid_smoothing(self.smoothing)

    @override
    def _compute(self, probabilities: Tensor, truths: Tensor) -> Tensor:
        true_positives, false_positives, false_negatives = self._count_overlaps(probabilities, truths)
        overlaps: Tensor = 2.0 * true_positives + self.smoothing
        return (1.0 - overlaps / (overlaps + false_positives + false_negatives)).mean()
