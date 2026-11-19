import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class GaussianMixtureNLLLoss(DistributionLoss):
    """
    Mixture density network likelihood; regression whose targets have several plausible values.

    Gaussian mixture negative log-likelihood (Bishop, 1994): for multimodal targets, such as trajectories that may
    turn either way or inverse problems, where a single Gaussian would predict the implausible average.

    The predictions hold ``(mixture logits, means, raw scales)`` on the last axis, ``components`` of each in that
    order; the weights are the softmax of the logits and each standard deviation is
    ``softplus(raw_scale) + minimum_scale``.

    Attributes
    ----------
    components : int
        Number of Gaussian components.
    minimum_scale : float
        Lower bound of each standard deviation, which keeps a component from collapsing onto one target.

    Raises
    ------
    ValueError
        If components is below 1 or minimum_scale is not positive.
    """

    components: int = 3
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.components < 1:
            raise ValueError(f"components must be at least 1, got {self.components}")
        if self.minimum_scale <= 0.0:
            raise ValueError(f"minimum_scale must be positive, got {self.minimum_scale}")

    @property
    @override
    def parameter_count(self) -> int:
        return 3 * self.components

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        logits, means, raw_scales = self._split(predictions)
        scales: Tensor = functional.softplus(raw_scales) + self.minimum_scale
        standardized: Tensor = (targets.unsqueeze(-1) - means) / scales
        component_log_densities: Tensor = -scales.log() - 0.5 * standardized.pow(2) - 0.5 * math.log(2.0 * math.pi)
        return -torch.logsumexp(functional.log_softmax(logits, dim=-1) + component_log_densities, dim=-1)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        logits, means, _ = self._split(predictions)
        return (functional.softmax(logits, dim=-1) * means).sum(dim=-1)

    def _split(self, predictions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Split the predictions into mixture logits, means and raw scales.
        """
        count: int = self.components
        return predictions[..., :count], predictions[..., count : 2 * count], predictions[..., 2 * count :]
