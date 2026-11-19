from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..distribution_loss import DistributionLoss
from .poisson import PoissonNLLLoss


@dataclass(frozen=True)
class ZeroInflatedPoissonNLLLoss(DistributionLoss):
    """
    Likelihood of counts with excess zeros, such as sparse sales or rare events, around a Poisson.

    Zero-inflated Poisson negative log-likelihood (Lambert, 1992): a structural zero with probability ``pi``,
    otherwise a Poisson count, for counts with more zeros than a Poisson allows.

    The predictions hold ``(log_rate, zero_logit)`` on the last axis; ``pi = sigmoid(zero_logit)``. The targets are
    non-negative counts.
    """

    @property
    @override
    def parameter_count(self) -> int:
        return 2

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        count_log_likelihoods: Tensor = -PoissonNLLLoss().compute_elementwise(predictions[..., :1], targets)
        return -self.mix_with_zeros(
            count_log_likelihoods, predictions[..., 0].exp().neg(), predictions[..., 1], targets
        )

    @override
    def _compute_mean(self, predictions: Tensor) -> Tensor:
        return torch.sigmoid(-predictions[..., 1]) * predictions[..., 0].exp()

    @staticmethod
    def mix_with_zeros(
        count_log_likelihoods: Tensor, zero_log_likelihoods: Tensor, zero_logits: Tensor, targets: Tensor
    ) -> Tensor:
        """
        Compute the log-likelihood of a count distribution inflated with structural zeros.

        Parameters
        ----------
        count_log_likelihoods : Tensor
            Log-likelihood of each target under the count distribution.
        zero_log_likelihoods : Tensor
            Log-likelihood of a zero under the count distribution.
        zero_logits : Tensor
            Logit of the probability of a structural zero.
        targets : Tensor
            The counts.

        Returns
        -------
        Tensor
            The log-likelihood of each target under the inflated distribution.
        """
        zero_log_probabilities: Tensor = functional.logsigmoid(zero_logits)
        count_log_probabilities: Tensor = functional.logsigmoid(-zero_logits)
        zeros: Tensor = torch.logaddexp(zero_log_probabilities, count_log_probabilities + zero_log_likelihoods)
        return torch.where(targets == 0.0, zeros, count_log_probabilities + count_log_likelihoods)
