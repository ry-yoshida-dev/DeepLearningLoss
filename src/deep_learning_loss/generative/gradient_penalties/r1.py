from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..gradient_penalty import GradientPenalty


@dataclass(frozen=True)
class R1GradientPenalty(GradientPenalty):
    """
    Zero-centered penalty on real samples; the standard regularizer of StyleGAN and modern GANs.

    R1 regularization (Mescheder et al., 2018): the squared gradient norm of the discriminator at real samples,
    which keeps it from pushing away from the data and makes training converge locally. Pass real samples as the
    inputs and weight the penalty by ``gamma / 2``.
    """

    @override
    def _compute(self, squared_norms: Tensor) -> Tensor:
        return squared_norms.mean()
