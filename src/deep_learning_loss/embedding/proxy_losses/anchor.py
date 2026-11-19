from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..proxy_loss import ProxyEmbeddingLoss


@dataclass(frozen=True)
class ProxyAnchorLoss(ProxyEmbeddingLoss):
    """
    Proxy loss that weighs each sample by its hardness; fast-converging metric learning for image retrieval.

    Proxy-Anchor loss (Kim et al., 2020): each proxy acts as an anchor that pulls the embeddings of its class and
    pushes the others through a log-sum-exp, so harder samples get larger gradients. The positive term averages over
    the classes present in the batch, the negative term over all classes.

    Attributes
    ----------
    margin : float
        Similarity margin that positives must exceed and negatives stay below.
    alpha : float
        Scaling of the similarities; larger values focus more on the hardest samples.

    Raises
    ------
    ValueError
        If margin is negative or alpha is not positive.
    """

    margin: float = 0.1
    alpha: float = 32.0

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @override
    def _compute(self, similarities: Tensor, labels: Tensor) -> Tensor:
        is_positive: Tensor = functional.one_hot(labels, similarities.shape[1]).bool()
        is_present: Tensor = is_positive.any(dim=0)
        # The lowest finite value drops a sample from a log-sum-exp without the NaN gradient of an infinite one.
        lowest: float = torch.finfo(similarities.dtype).min
        positive_exponents: Tensor = (-self.alpha * (similarities - self.margin)).masked_fill(~is_positive, lowest)
        negative_exponents: Tensor = (self.alpha * (similarities + self.margin)).masked_fill(is_positive, lowest)
        positive_terms: Tensor = functional.softplus(torch.logsumexp(positive_exponents, dim=0))
        negative_terms: Tensor = functional.softplus(torch.logsumexp(negative_exponents, dim=0))
        positive_loss: Tensor = torch.where(is_present, positive_terms, 0.0).sum() / is_present.sum().clamp(min=1)
        return positive_loss + negative_terms.mean()
