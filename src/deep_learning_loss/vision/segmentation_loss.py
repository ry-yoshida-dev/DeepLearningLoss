from abc import ABC, abstractmethod

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..loss import Loss


class SegmentationLoss(Loss, ABC):
    """
    Region loss between predicted class probabilities and the true class map, averaged over classes.

    Each class counts equally however few pixels it covers, which suits imbalanced segmentation; it is often added to
    a ``ClassificationLoss`` such as cross-entropy. The logits have shape ``(batch, classes, *spatial)`` and the
    indices ``(batch, *spatial)``. With a single logit channel the task is binary: the sigmoid gives the foreground
    probability, the indices are 0 or 1, and only the foreground region counts. The regions span the whole batch.
    """

    def compute(self, logits: Tensor, indices: Tensor, is_included: Tensor | None = None) -> Tensor:
        """
        Compute the loss averaged over classes.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores with shape ``(batch, classes, *spatial)``, or ``(batch, 1, *spatial)`` for
            binary segmentation.
        indices : Tensor
            True class indices with shape ``(batch, *spatial)``; indices outside the mask may be anything.
        is_included : Tensor | None
            Boolean mask of the shape of the indices marking the pixels that count. All pixels when None.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        TypeError
            If the indices are not integers or the mask is not boolean.
        ValueError
            If the shapes of the inputs do not match.
        """
        self._require_indices("indices", indices)
        if logits.ndim < 2 or logits.shape[:1] + logits.shape[2:] != indices.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} must be (batch, classes, *spatial) "
                + f"for indices shape {tuple(indices.shape)} (batch, *spatial)"
            )
        valid_indices: Tensor = indices.long()
        if is_included is not None:
            self._require_mask(is_included, "indices", indices)
            valid_indices = torch.where(is_included, valid_indices, 0)
        probabilities: Tensor
        truths: Tensor
        if logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits)
            truths = (valid_indices == 1).unsqueeze(1).to(logits.dtype)
        else:
            probabilities = functional.softmax(logits, dim=1)
            truths = functional.one_hot(valid_indices, logits.shape[1]).movedim(-1, 1).to(logits.dtype)
        if is_included is not None:
            weights: Tensor = is_included.unsqueeze(1).to(logits.dtype)
            probabilities = probabilities * weights
            truths = truths * weights
        # Classes first and every pixel of the batch flattened, so that each region spans the whole batch.
        return self._compute(probabilities.movedim(1, 0).flatten(1), truths.movedim(1, 0).flatten(1))

    @staticmethod
    def _require_valid_smoothing(smoothing: float) -> None:
        if smoothing < 0.0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")

    @staticmethod
    def _count_overlaps(probabilities: Tensor, truths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Count the soft true positives, false positives and false negatives of each class.
        """
        true_positives: Tensor = (probabilities * truths).sum(dim=1)
        false_positives: Tensor = probabilities.sum(dim=1) - true_positives
        false_negatives: Tensor = truths.sum(dim=1) - true_positives
        return true_positives, false_positives, false_negatives

    @abstractmethod
    def _compute(self, probabilities: Tensor, truths: Tensor) -> Tensor:
        """
        Compute the scalar loss from class probabilities and one-hot truths of shape ``(classes, pixels)``.

        Pixels outside the mask are zero in both.
        """
