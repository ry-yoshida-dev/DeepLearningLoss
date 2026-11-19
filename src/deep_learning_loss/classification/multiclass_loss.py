from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..loss import Loss


class ClassificationLoss(Loss, ABC):
    """
    Loss of class logits against the one true class index, for single-label multiclass classification.

    The logits have shape ``(batch, classes, *positions)`` and the indices ``(batch, *positions)``, so the same losses
    serve sequence labeling and per-pixel segmentation.
    """

    def compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        """
        Compute the loss at each position.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores with shape ``(batch, classes, *positions)``.
        indices : Tensor
            True class indices with shape ``(batch, *positions)``.

        Returns
        -------
        Tensor
            The loss at each position, with the shape of the indices.

        Raises
        ------
        TypeError
            If the indices are not integers.
        ValueError
            If the shapes of logits and indices do not match.
        """
        self._validate_inputs(logits, indices)
        return self._compute_elementwise(logits, indices.long())

    def compute(
        self, logits: Tensor, indices: Tensor, class_weights: Tensor | None = None, is_included: Tensor | None = None
    ) -> Tensor:
        """
        Compute the mean loss, optionally weighted by class and restricted to the positions a mask selects.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores with shape ``(batch, classes, *positions)``.
        indices : Tensor
            True class indices with shape ``(batch, *positions)``; indices outside the mask may be anything, such as
            an ignore label of 255.
        class_weights : Tensor | None
            Weight of each class with shape ``(classes,)``, against class imbalance; the mean is then weighted by the
            weight of each true class. Uniform when None.
        is_included : Tensor | None
            Boolean mask of the shape of the indices marking the positions to average. All positions when None.

        Returns
        -------
        Tensor
            The scalar mean loss; zero when nothing is selected.

        Raises
        ------
        TypeError
            If the indices are not integers or the mask is not boolean.
        ValueError
            If the shapes of the inputs do not match.
        """
        self._validate_inputs(logits, indices)
        if class_weights is not None and class_weights.shape != (logits.shape[1],):
            raise ValueError(f"class_weights must have shape ({logits.shape[1]},), got {tuple(class_weights.shape)}")
        valid_indices: Tensor = indices.long()
        if is_included is not None:
            self._require_mask(is_included, "indices", indices)
            valid_indices = torch.where(is_included, valid_indices, 0)
        losses: Tensor = self._compute_elementwise(logits, valid_indices)
        weights: Tensor = torch.ones_like(losses) if class_weights is None else class_weights[valid_indices]
        if is_included is not None:
            weights = weights * is_included
        return (weights * losses).sum() / weights.sum().clamp(min=torch.finfo(losses.dtype).tiny)

    def _validate_inputs(self, logits: Tensor, indices: Tensor) -> None:
        self._require_indices("indices", indices)
        if logits.ndim < 2 or logits.shape[:1] + logits.shape[2:] != indices.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} must be (batch, classes, *positions) "
                + f"for indices shape {tuple(indices.shape)} (batch, *positions)"
            )

    @abstractmethod
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor: ...
