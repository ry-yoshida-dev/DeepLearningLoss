from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..segmentation_loss import SegmentationLoss


@dataclass(frozen=True)
class LovaszSoftmaxLoss(SegmentationLoss):
    """
    Convex surrogate of the IoU; optimizes the benchmark metric directly, usually after pretraining with cross-entropy.

    Lovasz-Softmax loss (Berman et al., 2018): the Lovasz extension of the Jaccard loss applied to the pixel errors
    ``|truth - probability|`` of each class, a tight convex surrogate of the intersection over union; it often
    fine-tunes a network pretrained with cross-entropy.

    Classes absent from the target are skipped; a target without any class gives a zero loss.
    """

    @override
    def _compute(self, probabilities: Tensor, truths: Tensor) -> Tensor:
        errors: Tensor = (truths - probabilities).abs()
        sorted_errors, order = errors.sort(dim=1, descending=True)
        sorted_truths: Tensor = truths.gather(1, order)
        # Jaccard loss of each prefix of the pixels sorted by error; its increments weigh the sorted errors.
        foreground_totals: Tensor = sorted_truths.sum(dim=1, keepdim=True)
        intersections: Tensor = foreground_totals - sorted_truths.cumsum(dim=1)
        unions: Tensor = foreground_totals + (1.0 - sorted_truths).cumsum(dim=1)
        jaccard_losses: Tensor = 1.0 - intersections / unions
        increments: Tensor = torch.cat([jaccard_losses[:, :1], jaccard_losses.diff(dim=1)], dim=1)
        class_losses: Tensor = (sorted_errors * increments).sum(dim=1)
        is_present: Tensor = foreground_totals.squeeze(1) > 0.0
        return torch.where(is_present, class_losses, 0.0).sum() / is_present.sum().clamp(min=1)
