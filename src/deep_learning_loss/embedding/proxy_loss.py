from abc import ABC, abstractmethod

import torch.nn.functional as functional
from torch import Tensor

from ..loss import Loss


class ProxyEmbeddingLoss(Loss, ABC):
    """
    Loss of embeddings with class labels against one learnable proxy embedding per class, for metric learning.

    Each class is represented by a proxy, a learnable vector trained with the network (for example an
    ``nn.Parameter`` of shape ``(classes, features)``), so every batch compares its embeddings with all classes and
    needs no sampling of pairs; the usual choice for face recognition and retrieval with many classes. The losses
    compare cosine similarities, so neither embeddings nor proxies need to be normalized.
    """

    def compute(self, embeddings: Tensor, labels: Tensor, proxies: Tensor) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        embeddings : Tensor
            Embeddings with shape ``(batch, features)``.
        labels : Tensor
            Integer class labels in ``[0, classes)`` with shape ``(batch,)``.
        proxies : Tensor
            Learnable class proxies with shape ``(classes, features)``.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        TypeError
            If the labels are not integers.
        ValueError
            If the shapes do not match.
        """
        self._require_indices("labels", labels)
        if embeddings.ndim != 2 or labels.shape != embeddings.shape[:1]:
            raise ValueError(
                f"embeddings shape {tuple(embeddings.shape)} must be (batch, features) "
                + f"for labels shape {tuple(labels.shape)} (batch,)"
            )
        if proxies.ndim != 2 or proxies.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"proxies shape {tuple(proxies.shape)} must be (classes, features) "
                + f"for embeddings shape {tuple(embeddings.shape)}"
            )
        similarities: Tensor = functional.normalize(embeddings, dim=1) @ functional.normalize(proxies, dim=1).T
        return self._compute(similarities, labels.long())

    @abstractmethod
    def _compute(self, similarities: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the scalar loss from the cosine similarities of shape ``(batch, classes)`` and the labels.
        """
