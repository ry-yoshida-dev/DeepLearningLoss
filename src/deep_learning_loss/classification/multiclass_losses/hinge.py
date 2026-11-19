from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..multiclass_loss import ClassificationLoss


@dataclass(frozen=True)
class MulticlassHingeLoss(ClassificationLoss):
    """
    SVM-style margin loss on class scores.

    Multiclass hinge loss (Weston and Watkins, 1999): the mean over the wrong classes of how far each comes within a
    margin of the true class score, as in support vector machines; scores need not be probabilities.

    Attributes
    ----------
    margin : float
        Score gap by which the true class should beat every other class.

    Raises
    ------
    ValueError
        If margin is not positive.
    """

    margin: float = 1.0

    def __post_init__(self) -> None:
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")

    @override
    def _compute_elementwise(self, logits: Tensor, indices: Tensor) -> Tensor:
        true_logits: Tensor = logits.gather(1, indices.unsqueeze(1))
        violations: Tensor = (self.margin - true_logits + logits).clamp(min=0.0)
        is_wrong_class: Tensor = torch.ones_like(logits, dtype=torch.bool).scatter(1, indices.unsqueeze(1), False)
        return torch.where(is_wrong_class, violations, 0.0).sum(dim=1) / logits.shape[1]
