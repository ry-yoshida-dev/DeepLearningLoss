from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..labeled_loss import LabeledEmbeddingLoss


@dataclass(frozen=True)
class MultiSimilarityLoss(LabeledEmbeddingLoss):
    """
    Pair-weighting loss with hard-pair mining; strong metric learning for retrieval with many classes.

    Multi-Similarity loss (Wang et al., 2019): mines the informative pairs of each anchor, positives less similar
    than its hardest negative and negatives more similar than its hardest positive (by ``mining_margin``), then
    weighs them through soft-plus log-sum-exps of the cosine similarities around ``threshold``.

    Anchors without a mined positive and negative are skipped.

    Attributes
    ----------
    positive_scale : float
        Scaling of the positive similarities (alpha in the paper).
    negative_scale : float
        Scaling of the negative similarities (beta in the paper).
    threshold : float
        Cosine similarity separating the pulled positives from the pushed negatives (lambda in the paper).
    mining_margin : float
        Slack of the hard-pair mining (epsilon in the paper).

    Raises
    ------
    ValueError
        If a scale is not positive or mining_margin is negative.
    """

    positive_scale: float = 2.0
    negative_scale: float = 50.0
    threshold: float = 0.5
    mining_margin: float = 0.1

    def __post_init__(self) -> None:
        if self.positive_scale <= 0.0 or self.negative_scale <= 0.0:
            raise ValueError(f"scales must be positive, got {self.positive_scale} and {self.negative_scale}")
        if self.mining_margin < 0.0:
            raise ValueError(f"mining_margin must be non-negative, got {self.mining_margin}")

    @override
    def _compute(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        normalized: Tensor = functional.normalize(embeddings, dim=1)
        similarities: Tensor = normalized @ normalized.T
        is_same_label: Tensor = self._compute_same_label_mask(labels)
        is_positive: Tensor = is_same_label & self._compute_off_diagonal_mask(labels)
        is_negative: Tensor = ~is_same_label
        detached: Tensor = similarities.detach()
        hardest_negatives: Tensor = torch.where(is_negative, detached, float("-inf")).amax(dim=1, keepdim=True)
        hardest_positives: Tensor = torch.where(is_positive, detached, float("inf")).amin(dim=1, keepdim=True)
        is_mined_positive: Tensor = is_positive & (detached - self.mining_margin < hardest_negatives)
        is_mined_negative: Tensor = is_negative & (detached + self.mining_margin > hardest_positives)
        is_valid_anchor: Tensor = is_mined_positive.any(dim=1) & is_mined_negative.any(dim=1)
        lowest: float = torch.finfo(similarities.dtype).min
        positive_exponents: Tensor = -self.positive_scale * (similarities - self.threshold)
        negative_exponents: Tensor = self.negative_scale * (similarities - self.threshold)
        positive_terms: Tensor = functional.softplus(
            torch.logsumexp(positive_exponents.masked_fill(~is_mined_positive, lowest), dim=1)
        )
        negative_terms: Tensor = functional.softplus(
            torch.logsumexp(negative_exponents.masked_fill(~is_mined_negative, lowest), dim=1)
        )
        losses: Tensor = positive_terms / self.positive_scale + negative_terms / self.negative_scale
        return torch.where(is_valid_anchor, losses, 0.0).sum() / is_valid_anchor.sum().clamp(min=1)
