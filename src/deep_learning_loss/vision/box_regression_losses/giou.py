from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..box_regression_loss import BoxRegressionLoss


@dataclass(frozen=True)
class GIoULoss(BoxRegressionLoss):
    """
    Default box regression loss; IoU loss with a gradient even for boxes that do not overlap.

    Generalized IoU loss (Rezatofighi et al., 2019) ``1 - IoU + (C - U) / C``, with ``C`` the area of the smallest box
    enclosing both and ``U`` their union: it keeps pulling a predicted box toward a target it does not overlap; the
    usual choice in DETR-style detectors.
    """

    @override
    def _compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        intersections, unions = self._compute_intersections_and_unions(predictions, targets)
        enclosing_areas: Tensor = self._compute_areas(self._compute_enclosing_boxes(predictions, targets))
        ious: Tensor = intersections / (unions + self._EPSILON)
        return 1.0 - ious + (enclosing_areas - unions) / (enclosing_areas + self._EPSILON)
