from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..loss import Loss


class LabeledEmbeddingLoss(Loss, ABC):
    """
    Loss of a batch of embeddings and their class labels, for supervised metric learning.

    The losses pull embeddings of the same label together and push those of different labels apart, comparing every
    pair in the batch, so batches should hold several samples of each label. The embeddings have shape
    ``(batch, features)`` and the labels ``(batch,)``. A batch without a usable pair gives a zero loss.
    """

    def compute(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        embeddings : Tensor
            Embeddings with shape ``(batch, features)``.
        labels : Tensor
            Integer class labels with shape ``(batch,)``.

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
        return self._compute(embeddings, labels)

    @staticmethod
    def _compute_distances(embeddings: Tensor) -> Tensor:
        squared_norms: Tensor = embeddings.pow(2).sum(dim=1)
        squared_distances: Tensor = (
            squared_norms.unsqueeze(1) + squared_norms.unsqueeze(0) - 2.0 * embeddings @ embeddings.T
        )
        # The floor keeps the gradient of the square root finite for coinciding embeddings.
        return squared_distances.clamp(min=1e-12).sqrt()

    @staticmethod
    def _compute_same_label_mask(labels: Tensor) -> Tensor:
        return labels.unsqueeze(1) == labels.unsqueeze(0)

    @staticmethod
    def _compute_off_diagonal_mask(labels: Tensor) -> Tensor:
        return ~torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)

    @abstractmethod
    def _compute(self, embeddings: Tensor, labels: Tensor) -> Tensor: ...
