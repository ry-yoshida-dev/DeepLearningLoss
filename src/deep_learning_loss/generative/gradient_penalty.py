from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..loss import Loss


class GradientPenalty(Loss, ABC):
    """
    Penalty on the gradient of the discriminator with respect to its inputs, which stabilizes GAN training.

    It is added, with a weight, to the discriminator loss of an ``AdversarialLoss``. The inputs must require
    gradients and the scores must be the discriminator outputs computed from them, so that the penalty can
    differentiate the scores through the discriminator.
    """

    def compute(self, inputs: Tensor, scores: Tensor) -> Tensor:
        """
        Compute the mean penalty over the samples.

        Parameters
        ----------
        inputs : Tensor
            Discriminator inputs with shape ``(batch, *features)`` that require gradients.
        scores : Tensor
            Discriminator outputs computed from the inputs, with shape ``(batch, *any)``.

        Returns
        -------
        Tensor
            The scalar penalty, differentiable with respect to the discriminator parameters.

        Raises
        ------
        ValueError
            If the inputs do not require gradients or the batch sizes differ.
        """
        if not inputs.requires_grad:
            raise ValueError("inputs must require gradients; call inputs.requires_grad_() before the discriminator")
        if inputs.ndim == 0 or scores.ndim == 0 or inputs.shape[0] != scores.shape[0]:
            raise ValueError(
                f"inputs shape {tuple(inputs.shape)} and scores shape {tuple(scores.shape)} must share a batch axis"
            )
        (gradients,) = torch.autograd.grad(scores.sum(), inputs, create_graph=True)
        return self._compute(gradients.flatten(1).pow(2).sum(dim=1))

    @abstractmethod
    def _compute(self, squared_norms: Tensor) -> Tensor:
        """
        Compute the scalar penalty from the squared gradient norm of each sample.
        """
