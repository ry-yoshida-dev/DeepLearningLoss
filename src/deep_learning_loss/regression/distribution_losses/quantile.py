from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..distribution_loss import DistributionLoss


@dataclass(frozen=True)
class QuantileLoss(DistributionLoss):
    """
    Pinball loss averaged over several quantile levels; distribution-free forecasts and prediction intervals.

    Multi-quantile loss (as in MQ-RNN and the Temporal Fusion Transformer): the pinball loss of each predicted
    quantile at its level, averaged over the levels. It describes the distribution by its quantiles without assuming a
    shape, and approximates the CRPS as the levels grow dense.

    The predictions hold one quantile per level on the last axis, in the order of ``levels``. Nothing keeps the
    predicted quantiles from crossing; predict increments with a softplus and sum them when monotonicity matters. The
    mean is approximated by integrating the quantile function, each quantile covering the levels closer to its own
    than to its neighbors'.

    Attributes
    ----------
    levels : tuple[float, ...]
        Strictly increasing quantile levels in (0, 1).

    Raises
    ------
    ValueError
        If levels is empty, not strictly increasing, or outside (0, 1).
    """

    levels: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("levels must not be empty")
        if not all(0.0 < level < 1.0 for level in self.levels):
            raise ValueError(f"levels must be in (0, 1), got {self.levels}")
        if any(lower >= upper for lower, upper in zip(self.levels, self.levels[1:], strict=False)):
            raise ValueError(f"levels must be strictly increasing, got {self.levels}")

    @property
    @override
    def parameter_count(self) -> int:
        return len(self.levels)

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        levels: Tensor = torch.tensor(self.levels, dtype=predictions.dtype, device=predictions.device)
        errors: Tensor = targets.unsqueeze(-1) - predictions
        return torch.maximum(levels * errors, (levels - 1.0) * errors).mean(dim=-1)

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        midpoints: list[float] = [
            (lower + upper) / 2.0 for lower, upper in zip(self.levels, self.levels[1:], strict=False)
        ]
        boundaries: Tensor = torch.tensor([0.0, *midpoints, 1.0], dtype=predictions.dtype, device=predictions.device)
        return predictions @ boundaries.diff()
