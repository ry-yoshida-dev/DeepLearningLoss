from abc import ABC, abstractmethod

from torch import Tensor

from ..elementwise_loss import ElementwiseLoss


class DistributionLoss(ElementwiseLoss, ABC):
    """
    Proper scoring rule, usually the negative log-likelihood, of the targets under a predicted distribution.

    Use it when the network should express its uncertainty, or when the targets are counts or bounded. The network
    predicts the parameters of a distribution for each target value, or a set of its quantiles, stacked unconstrained
    on a last axis of size ``parameter_count``, so their shape is the shape of the targets plus that axis; each loss
    maps them to valid parameters itself, so a linear output layer suffices.
    """

    @property
    @abstractmethod
    def parameter_count(self) -> int:
        """
        Number of distribution parameters predicted for each target value.
        """

    def compute_mean(self, predictions: Tensor) -> Tensor:
        """
        Compute the mean of each predicted distribution, the point prediction that the parameters imply.

        Parameters
        ----------
        predictions : Tensor
            The unconstrained parameters, with a last axis of size ``parameter_count``.

        Returns
        -------
        Tensor
            The mean of each distribution, with the shape of the predictions without the last axis.

        Raises
        ------
        ValueError
            If the last axis of the predictions is not of size ``parameter_count``.
        """
        self._require_parameter_axis(predictions)
        return self._compute_mean(predictions)

    def _validate_shapes(self, predictions: Tensor, targets: Tensor) -> None:
        self._require_parameter_axis(predictions)
        if predictions.shape[:-1] != targets.shape:
            raise ValueError(
                f"predictions shape {tuple(predictions.shape)} must be the targets shape {tuple(targets.shape)} "
                + f"plus a last axis of size {self.parameter_count}"
            )

    def _require_parameter_axis(self, predictions: Tensor) -> None:
        if predictions.ndim == 0 or predictions.shape[-1] != self.parameter_count:
            raise ValueError(
                f"predictions must have a last axis of size {self.parameter_count}, "
                + f"got shape {tuple(predictions.shape)}"
            )

    @abstractmethod
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the score of each target under the distribution its parameters describe; lower is better.
        """

    @abstractmethod
    def _compute_mean(self, predictions: Tensor) -> Tensor: ...
