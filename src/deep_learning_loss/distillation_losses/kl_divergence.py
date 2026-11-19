from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..distillation_loss import DistillationLoss


@dataclass(frozen=True)
class KLDivergenceLoss(DistillationLoss):
    """
    Default distillation loss; the student covers every class the teacher deems plausible.

    Kullback-Leibler divergence from the teacher to the student (Hinton et al., 2015): the default distillation loss;
    the student covers every class the teacher deems plausible.
    """

    @override
    def _compute_divergences(self, student_log_probabilities: Tensor, teacher_log_probabilities: Tensor) -> Tensor:
        teacher_probabilities: Tensor = teacher_log_probabilities.exp()
        return (teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)).sum(dim=1)
