from abc import ABC, abstractmethod

from torch import Tensor

from ..elementwise_loss import ElementwiseLoss


class BinaryClassificationLoss(ElementwiseLoss, ABC):
    """
    Loss of independent yes/no logits against targets of the same shape, for binary and multi-label classification.

    The predictions are logits (before the sigmoid), and the targets are probabilities of the positive class: 0 or 1,
    or values in between for soft labels.
    """

    @abstractmethod
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss of each logit against the target probability at the same position.
        """
