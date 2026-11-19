from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .loss import Loss


class ElementwiseLoss(Loss, ABC):
    """
    Loss defined value by value, averaged over all values or over the values a mask selects.
    """

    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss of each target value.

        Parameters
        ----------
        predictions : Tensor
            The network outputs; see the family for their expected shape and meaning.
        targets : Tensor
            The targets.

        Returns
        -------
        Tensor
            The loss of each value: the shape of the targets, without the last axis when a family stores each value
            along it, such as the four coordinates of a box.

        Raises
        ------
        ValueError
            If the shapes of predictions and targets do not match.
        """
        self._validate_shapes(predictions, targets)
        return self._compute_elementwise(predictions, targets)

    def compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean loss over all values.

        Parameters
        ----------
        predictions : Tensor
            The network outputs.
        targets : Tensor
            The targets.

        Returns
        -------
        Tensor
            The scalar mean loss.
        """
        return self.compute_elementwise(predictions, targets).mean()

    def compute_masked(self, predictions: Tensor, targets: Tensor, is_included: Tensor) -> Tensor:
        """
        Compute the mean loss over the values a mask selects, such as observed values or unpadded steps.

        Parameters
        ----------
        predictions : Tensor
            The network outputs.
        targets : Tensor
            The targets; values outside the mask may be anything, including NaN.
        is_included : Tensor
            Boolean mask of the shape of ``compute_elementwise``'s result marking the values to average.

        Returns
        -------
        Tensor
            The scalar mean loss; zero when the mask selects nothing.

        Raises
        ------
        TypeError
            If is_included is not boolean.
        ValueError
            If the shapes of predictions and targets do not match, or is_included does not have the shape of the
            values.
        """
        self._validate_shapes(predictions, targets)
        if is_included.dtype != torch.bool:
            raise TypeError(f"is_included must be a boolean tensor, got {is_included.dtype}")
        value_shape: tuple[int, ...] = self._value_shape(targets)
        if is_included.shape != value_shape:
            raise ValueError(f"is_included shape {tuple(is_included.shape)} differs from value shape {value_shape}")
        # Excluded targets are replaced by valid placeholders first, so that NaN or out-of-support values cannot leak
        # NaN into the gradients.
        target_mask: Tensor = is_included.reshape(value_shape + (1,) * (targets.ndim - is_included.ndim))
        included_targets: Tensor = torch.where(target_mask, targets, self._placeholder_targets(targets))
        losses: Tensor = self._compute_elementwise(predictions, included_targets)
        return torch.where(is_included, losses, 0.0).sum() / is_included.sum().clamp(min=1)

    def _validate_shapes(self, predictions: Tensor, targets: Tensor) -> None:
        self._require_same_shape("predictions", predictions, "targets", targets)

    def _value_shape(self, targets: Tensor) -> tuple[int, ...]:
        return tuple(targets.shape)

    def _placeholder_targets(self, targets: Tensor) -> Tensor:
        # Any target inside the support of every loss of the family; families with a narrower support override it.
        return torch.zeros_like(targets)

    @abstractmethod
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor: ...
