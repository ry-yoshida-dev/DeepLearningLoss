import math
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..distillation_loss import DistillationLoss


@dataclass(frozen=True)
class JensenShannonDivergenceLoss(DistillationLoss):
    """
    Symmetric, bounded distillation loss; also suits mutual learning between peers.

    Jensen-Shannon divergence between the student and the teacher: symmetric and bounded by log 2, a compromise
    between the forward and reverse KL divergences that also suits mutual learning between peers.
    """

    @override
    def _compute_divergences(self, student_log_probabilities: Tensor, teacher_log_probabilities: Tensor) -> Tensor:
        mixture_log_probabilities: Tensor = torch.logaddexp(
            student_log_probabilities, teacher_log_probabilities
        ) - math.log(2.0)
        student_terms: Tensor = student_log_probabilities.exp() * (
            student_log_probabilities - mixture_log_probabilities
        )
        teacher_terms: Tensor = teacher_log_probabilities.exp() * (
            teacher_log_probabilities - mixture_log_probabilities
        )
        return 0.5 * (student_terms + teacher_terms).sum(dim=1)
