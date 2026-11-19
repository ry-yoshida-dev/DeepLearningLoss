from .cross_entropy import CrossEntropyLoss
from .focal import FocalLoss
from .generalized_cross_entropy import GeneralizedCrossEntropyLoss
from .hinge import MulticlassHingeLoss
from .poly1 import Poly1Loss

__all__ = ["CrossEntropyLoss", "FocalLoss", "GeneralizedCrossEntropyLoss", "MulticlassHingeLoss", "Poly1Loss"]
