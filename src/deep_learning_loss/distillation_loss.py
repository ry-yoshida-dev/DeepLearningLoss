from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor

from .loss import Loss


@dataclass(frozen=True)
class DistillationLoss(Loss, ABC):
    """
    Divergence between the class distributions of a student and a teacher, for knowledge distillation.

    Both take logits of shape ``(batch, classes, *positions)``, softened by the temperature. The loss is scaled by the
    squared temperature (Hinton et al., 2015), so its gradients keep their size as the temperature changes. The
    teacher logits are not detached, so detach them for a frozen teacher.

    Attributes
    ----------
    temperature : float
        Divides the logits before the softmax; above 1 reveals how the teacher ranks the wrong classes.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def compute(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """
        Compute the mean divergence over the batch and positions.

        Parameters
        ----------
        student_logits : Tensor
            Logits of the student with shape ``(batch, classes, *positions)``.
        teacher_logits : Tensor
            Logits of the teacher, with the shape of the student logits.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        ValueError
            If the shapes differ or have no class axis.
        """
        self._require_same_shape("student_logits", student_logits, "teacher_logits", teacher_logits)
        if student_logits.ndim < 2:
            raise ValueError(f"logits must have shape (batch, classes, *positions), got {tuple(student_logits.shape)}")
        student_log_probabilities: Tensor = functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_log_probabilities: Tensor = functional.log_softmax(teacher_logits / self.temperature, dim=1)
        divergences: Tensor = self._compute_divergences(student_log_probabilities, teacher_log_probabilities)
        return divergences.mean() * self.temperature**2

    @abstractmethod
    def _compute_divergences(self, student_log_probabilities: Tensor, teacher_log_probabilities: Tensor) -> Tensor:
        """
        Compute the divergence at each position from log-probabilities over axis 1, removing that axis.
        """
