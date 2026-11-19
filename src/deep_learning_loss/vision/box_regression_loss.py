import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..elementwise_loss import ElementwiseLoss


class BoxRegressionLoss(ElementwiseLoss, ABC):
    """
    Overlap loss between predicted and target bounding boxes, for object detection and tracking.

    The boxes hold ``(x_min, y_min, x_max, y_max)`` on a last axis of size 4, the predictions already decoded from
    the network outputs; the loss of each box is one value, so masks select boxes, such as the anchors matched to an
    object. Being based on the intersection over union, the losses do not depend on the scale of the boxes.
    """

    _EPSILON: float = 1e-7

    def _compute_ious(self, predictions: Tensor, targets: Tensor) -> Tensor:
        intersections, unions = self._compute_intersections_and_unions(predictions, targets)
        return intersections / (unions + self._EPSILON)

    def _compute_intersections_and_unions(self, predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        intersection_widths: Tensor = (
            torch.minimum(predictions[..., 2], targets[..., 2]) - torch.maximum(predictions[..., 0], targets[..., 0])
        ).clamp(min=0.0)
        intersection_heights: Tensor = (
            torch.minimum(predictions[..., 3], targets[..., 3]) - torch.maximum(predictions[..., 1], targets[..., 1])
        ).clamp(min=0.0)
        intersections: Tensor = intersection_widths * intersection_heights
        unions: Tensor = self._compute_areas(predictions) + self._compute_areas(targets) - intersections
        return intersections, unions

    @staticmethod
    def _compute_areas(boxes: Tensor) -> Tensor:
        return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)

    @staticmethod
    def _compute_enclosing_boxes(predictions: Tensor, targets: Tensor) -> Tensor:
        return torch.cat(
            [
                torch.minimum(predictions[..., :2], targets[..., :2]),
                torch.maximum(predictions[..., 2:], targets[..., 2:]),
            ],
            dim=-1,
        )

    def _compute_center_penalties(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the squared distance between the box centers over the squared diagonal of the enclosing box.
        """
        center_gaps: Tensor = (predictions[..., :2] + predictions[..., 2:] - targets[..., :2] - targets[..., 2:]) / 2.0
        enclosing_boxes: Tensor = self._compute_enclosing_boxes(predictions, targets)
        diagonals: Tensor = (enclosing_boxes[..., 2:] - enclosing_boxes[..., :2]).pow(2).sum(dim=-1)
        return center_gaps.pow(2).sum(dim=-1) / (diagonals + self._EPSILON)

    def _compute_aspect_penalties(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the squared gap between the aspect-ratio angles of the boxes, scaled to [0, 1].
        """
        prediction_angles: Tensor = torch.atan(
            (predictions[..., 2] - predictions[..., 0]) / (predictions[..., 3] - predictions[..., 1] + self._EPSILON)
        )
        target_angles: Tensor = torch.atan(
            (targets[..., 2] - targets[..., 0]) / (targets[..., 3] - targets[..., 1] + self._EPSILON)
        )
        return 4.0 / math.pi**2 * (target_angles - prediction_angles).pow(2)

    def _validate_shapes(self, predictions: Tensor, targets: Tensor) -> None:
        self._require_same_shape("predictions", predictions, "targets", targets)
        if predictions.ndim == 0 or predictions.shape[-1] != 4:
            raise ValueError(f"boxes must have a last axis of size 4, got shape {tuple(predictions.shape)}")

    def _value_shape(self, targets: Tensor) -> tuple[int, ...]:
        return tuple(targets.shape[:-1])

    def _placeholder_targets(self, targets: Tensor) -> Tensor:
        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=targets.dtype, device=targets.device).expand_as(targets)

    @abstractmethod
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss of each predicted box against the target box at the same position.
        """
