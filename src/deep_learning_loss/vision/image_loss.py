from abc import ABC, abstractmethod

from torch import Tensor

from ..loss import Loss


class ImageLoss(Loss, ABC):
    """
    Loss between predicted and target images that compares local structure rather than independent pixels.

    Use it for image restoration, super-resolution, depth estimation and frame prediction, usually added to a pixel
    loss such as ``MAELoss``: pixel losses alone blur edges and textures that these losses keep. The images have shape
    ``(batch, channels, height, width)``.
    """

    def compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean loss over the images.

        Parameters
        ----------
        predictions : Tensor
            Predicted images with shape ``(batch, channels, height, width)``.
        targets : Tensor
            Target images with the shape of the predictions.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        ValueError
            If the shapes differ or are not four-dimensional, or the images are too small for the loss.
        """
        self._require_same_shape("predictions", predictions, "targets", targets)
        if predictions.ndim != 4:
            raise ValueError(f"images must have shape (batch, channels, height, width), got {tuple(predictions.shape)}")
        self._validate_size(predictions.shape[2], predictions.shape[3])
        return self._compute(predictions, targets)

    def _validate_size(self, height: int, width: int) -> None:
        if height < 2 or width < 2:
            raise ValueError(f"images must be at least 2 x 2, got {height} x {width}")

    @abstractmethod
    def _compute(self, predictions: Tensor, targets: Tensor) -> Tensor: ...
