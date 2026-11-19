from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..segmentation_loss import SegmentationLoss


@dataclass(frozen=True)
class JaccardLoss(SegmentationLoss):
    """
    Soft IoU loss; optimizes the intersection over union that segmentation benchmarks report.

    Soft Jaccard (IoU) loss ``1 - TP / (TP + FP + FN)``: optimizes the intersection over union that segmentation
    benchmarks report.

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
        overlaps: Tensor = true_positives + self.smoothing
        return (1.0 - overlaps / (overlaps + false_positives + false_negatives)).mean()
