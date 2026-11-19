from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..paired_loss import PairedEmbeddingLoss


@dataclass(frozen=True)
class NegativeCosineSimilarityLoss(PairedEmbeddingLoss):
    """
    Negative cosine similarity of pairs; the BYOL and SimSiam loss, with a predictor and stop-gradient.

    Negative cosine similarity of the paired rows, in [-1, 1]: the loss of BYOL and SimSiam, where the first views come
    from a predictor and the second from a target branch that must be detached (or a momentum encoder) to avoid
    collapse.
    """

    @property
    @override
    def _minimum_batch_size(self) -> int:
        return 1

    @override
    def _compute(self, first: Tensor, second: Tensor) -> Tensor:
        return -functional.cosine_similarity(first, second, dim=1).mean()
