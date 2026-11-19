from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..labeled_loss import LabeledEmbeddingLoss


@dataclass(frozen=True)
class SupervisedContrastiveLoss(LabeledEmbeddingLoss):
    """
    Default metric learning loss; contrasts each embedding with all others of its label.

    Supervised contrastive loss (Khosla et al., 2020): InfoNCE where every other embedding of the same label is a
    positive; the default for metric learning, and a strong pre-training objective for classifiers.

    Embeddings are compared by cosine similarity, and anchors without another embedding of their label are skipped.

    Attributes
    ----------
    temperature : float
        Divides the cosine similarities; smaller values concentrate on the hardest negatives.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @override
    def _compute(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        normalized: Tensor = functional.normalize(embeddings, dim=1)
        is_other: Tensor = self._compute_off_diagonal_mask(labels)
        is_positive: Tensor = self._compute_same_label_mask(labels) & is_other
        positive_counts: Tensor = is_positive.sum(dim=1)
        is_valid_anchor: Tensor = positive_counts > 0
        # A finite floor instead of -inf keeps the gradients finite for an anchor alone in its batch.
        logits: Tensor = (normalized @ normalized.T / self.temperature).masked_fill(
            ~is_other, torch.finfo(embeddings.dtype).min
        )
        log_probabilities: Tensor = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_sums: Tensor = torch.where(is_positive, log_probabilities, 0.0).sum(dim=1)
        anchor_losses: Tensor = -positive_sums / positive_counts.clamp(min=1)
        return torch.where(is_valid_anchor, anchor_losses, 0.0).sum() / is_valid_anchor.sum().clamp(min=1)
