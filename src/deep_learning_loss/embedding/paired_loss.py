from abc import ABC, abstractmethod

from torch import Tensor

from ..loss import Loss


class PairedEmbeddingLoss(Loss, ABC):
    """
    Loss of two batches of embeddings whose rows correspond, for self-supervised and cross-modal learning.

    Row ``i`` of the first batch and row ``i`` of the second are two views of the same sample (two augmentations, or
    an image and its caption); the losses pull them together and, depending on the loss, keep the representation from
    collapsing. Both batches have shape ``(batch, features)``.
    """

    def compute(self, first: Tensor, second: Tensor) -> Tensor:
        """
        Compute the loss of a batch of pairs.

        Parameters
        ----------
        first : Tensor
            Embeddings of the first views with shape ``(batch, features)``.
        second : Tensor
            Embeddings of the second views, with the shape of the first.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        ValueError
            If the shapes differ, are not two-dimensional, or hold fewer pairs than the loss needs.
        """
        self._require_same_shape("first", first, "second", second)
        if first.ndim != 2:
            raise ValueError(f"embeddings must have shape (batch, features), got {tuple(first.shape)}")
        if first.shape[0] < self._minimum_batch_size:
            raise ValueError(
                f"{type(self).__name__} needs at least {self._minimum_batch_size} pairs, got {first.shape[0]}"
            )
        return self._compute(first, second)

    @property
    def _minimum_batch_size(self) -> int:
        # Most paired losses compare each pair against the others in the batch.
        return 2

    @abstractmethod
    def _compute(self, first: Tensor, second: Tensor) -> Tensor: ...
