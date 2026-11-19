from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..gradient_penalty import GradientPenalty


@dataclass(frozen=True)
class WassersteinGradientPenalty(GradientPenalty):
    """
    One-centered penalty on interpolated samples; the WGAN-GP constraint that keeps the critic 1-Lipschitz.

    WGAN-GP penalty (Gulrajani et al., 2017): ``(||gradient|| - 1) ** 2`` at random interpolations between real and
    generated samples, made by ``interpolate``; pair it with ``WassersteinAdversarialLoss`` and a weight of about 10.
    """

    @override
    def _compute(self, squared_norms: Tensor) -> Tensor:
        # The floor keeps the gradient of the norm finite where the critic is flat.
        return (squared_norms.clamp(min=1e-12).sqrt() - 1.0).pow(2).mean()

    @staticmethod
    def interpolate(real: Tensor, fake: Tensor) -> Tensor:
        """
        Mix each real sample with a generated one at a uniformly random position along the line between them.

        Parameters
        ----------
        real : Tensor
            Real samples with shape ``(batch, *features)``.
        fake : Tensor
            Generated samples with the shape of the real ones.

        Returns
        -------
        Tensor
            The interpolated samples, detached and requiring gradients, ready for the critic.

        Raises
        ------
        ValueError
            If the shapes differ or lack a batch axis.
        """
        if real.shape != fake.shape or real.ndim == 0:
            raise ValueError(f"real shape {tuple(real.shape)} and fake shape {tuple(fake.shape)} must match")
        positions: Tensor = torch.rand((real.shape[0],) + (1,) * (real.ndim - 1), dtype=real.dtype, device=real.device)
        return torch.lerp(real.detach(), fake.detach(), positions).requires_grad_()
