from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..distillation_loss import DistillationLoss


@dataclass(frozen=True)
class ReverseKLDivergenceLoss(DistillationLoss):
    """
    Mode-seeking distillation; a small student concentrates on the teacher's main modes.

    Kullback-Leibler divergence from the student to the teacher: mode-seeking, so a small student concentrates on the
    teacher's main modes instead of spreading over all of them, as in distilling language models (Gu et al., 2024).
    """

    @override
    def _compute_divergences(self, student_log_probabilities: Tensor, teacher_log_probabilities: Tensor) -> Tensor:
        student_probabilities: Tensor = student_log_probabilities.exp()
        return (student_probabilities * (student_log_probabilities - teacher_log_probabilities)).sum(dim=1)
