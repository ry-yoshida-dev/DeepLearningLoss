from .hinge import HingeAdversarialLoss
from .least_squares import LeastSquaresAdversarialLoss
from .non_saturating import NonSaturatingAdversarialLoss
from .wasserstein import WassersteinAdversarialLoss

__all__ = [
    "HingeAdversarialLoss",
    "LeastSquaresAdversarialLoss",
    "NonSaturatingAdversarialLoss",
    "WassersteinAdversarialLoss",
]
