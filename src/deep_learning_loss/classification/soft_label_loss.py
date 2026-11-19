from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..loss import Loss


class SoftLabelClassificationLoss(Loss, ABC):
    """
    Loss of class logits against a target probability distribution over the classes, for mixed or uncertain labels.

    Use it when a sample belongs to several classes in proportion, as with Mixup and CutMix, annotator disagreement or
    pseudo-labels. The logits and the target probabilities share the shape ``(batch, classes, *positions)``; each
    target sums to 1 over the classes.
    """

    def compute_elementwise(self, logits: Tensor, probabilities: Tensor) -> Tensor:
        """
        Compute the loss at each position.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores with shape ``(batch, classes, *positions)``.
        probabilities : Tensor
            Target class probabilities with the shape of the logits.

        Returns
        -------
        Tensor
            The loss at each position, with shape ``(batch, *positions)``.

        Raises
        ------
        ValueError
            If the shapes differ or lack a class axis.
        """
        self._validate_inputs(logits, probabilities)
        return self._compute_elementwise(logits, probabilities)

    def compute(self, logits: Tensor, probabilities: Tensor, is_included: Tensor | None = None) -> Tensor:
        """
        Compute the mean loss, optionally restricted to the positions a mask selects.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores with shape ``(batch, classes, *positions)``.
        probabilities : Tensor
            Target class probabilities with the shape of the logits; values outside the mask may be anything,
            including NaN.
        is_included : Tensor | None
            Boolean mask with shape ``(batch, *positions)`` marking the positions to average. All positions when None.

        Returns
        -------
        Tensor
            The scalar mean loss; zero when nothing is selected.

        Raises
        ------
        TypeError
            If the mask is not boolean.
        ValueError
            If the shapes of the inputs do not match.
        """
        self._validate_inputs(logits, probabilities)
        if is_included is None:
            return self._compute_elementwise(logits, probabilities).mean()
        if is_included.dtype != torch.bool:
            raise TypeError(f"is_included must be a boolean tensor, got {is_included.dtype}")
        if is_included.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"is_included shape {tuple(is_included.shape)} must be (batch, *positions) "
                + f"for logits shape {tuple(logits.shape)}"
            )
        valid_probabilities: Tensor = torch.where(is_included.unsqueeze(1), probabilities, 0.0)
        losses: Tensor = self._compute_elementwise(logits, valid_probabilities)
        return torch.where(is_included, losses, 0.0).sum() / is_included.sum().clamp(min=1)

    def _validate_inputs(self, logits: Tensor, probabilities: Tensor) -> None:
        self._require_same_shape("logits", logits, "probabilities", probabilities)
        if logits.ndim < 2:
            raise ValueError(f"logits must have shape (batch, classes, *positions), got {tuple(logits.shape)}")

    @abstractmethod
    def _compute_elementwise(self, logits: Tensor, probabilities: Tensor) -> Tensor: ...
