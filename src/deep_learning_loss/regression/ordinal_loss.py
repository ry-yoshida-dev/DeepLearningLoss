from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..loss import Loss


class OrdinalRegressionLoss(Loss, ABC):
    """
    Loss of threshold logits against ordered class indices, for ratings, grades and severity levels.

    Ordinal regression predicts one of ``ranks`` ordered classes, such as an age group, a star rating or a disease
    stage, where a near miss is better than a far one; unlike classification it uses the order, and unlike regression
    it assumes no distance between the ranks. The network outputs ``ranks - 1`` logits, one per threshold ``k``,
    with shape ``(batch, ranks - 1, *positions)``; the indices have shape ``(batch, *positions)`` and lie in
    ``[0, ranks)``.
    """

    def compute(self, logits: Tensor, indices: Tensor) -> Tensor:
        """
        Compute the mean loss.

        Parameters
        ----------
        logits : Tensor
            Threshold logits with shape ``(batch, ranks - 1, *positions)``.
        indices : Tensor
            True ranks with shape ``(batch, *positions)``.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        TypeError
            If the indices are not integers.
        ValueError
            If the shapes do not match.
        """
        self._validate_inputs(logits, indices)
        thresholds: Tensor = torch.arange(logits.shape[1], device=logits.device).view(
            (1, -1) + (1,) * (logits.ndim - 2)
        )
        # Whether each rank exceeds each threshold: the binary target of each logit.
        is_above: Tensor = indices.long().unsqueeze(1) > thresholds
        return self._compute(logits, is_above)

    def compute_ranks(self, logits: Tensor) -> Tensor:
        """
        Predict the rank that the threshold logits imply.

        Parameters
        ----------
        logits : Tensor
            Threshold logits with shape ``(batch, ranks - 1, *positions)``.

        Returns
        -------
        Tensor
            Integer ranks in ``[0, ranks)`` with shape ``(batch, *positions)``.

        Raises
        ------
        ValueError
            If the logits lack a threshold axis.
        """
        if logits.ndim < 2:
            raise ValueError(f"logits must have shape (batch, ranks - 1, *positions), got {tuple(logits.shape)}")
        return (self._compute_exceedance_probabilities(logits) > 0.5).sum(dim=1)

    def _validate_inputs(self, logits: Tensor, indices: Tensor) -> None:
        self._require_indices("indices", indices)
        if logits.ndim < 2 or logits.shape[:1] + logits.shape[2:] != indices.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} must be (batch, ranks - 1, *positions) "
                + f"for indices shape {tuple(indices.shape)} (batch, *positions)"
            )

    @abstractmethod
    def _compute(self, logits: Tensor, is_above: Tensor) -> Tensor:
        """
        Compute the scalar loss from the logits and whether each rank exceeds each threshold.
        """

    @abstractmethod
    def _compute_exceedance_probabilities(self, logits: Tensor) -> Tensor:
        """
        Compute the probability that the rank exceeds each threshold.
        """
