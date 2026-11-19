from abc import ABC, abstractmethod

from torch import Tensor

from ..elementwise_loss import ElementwiseLoss


class PointLoss(ElementwiseLoss, ABC):
    """
    Loss between point predictions and targets of the same shape, defined by the error of each value.

    Use it for regression, forecasting, reconstruction and any task whose network outputs the target values directly.
    """

    @abstractmethod
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the non-negative loss of each predicted value against the target value at the same position.
        """
