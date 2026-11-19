from .labeled_loss import LabeledEmbeddingLoss
from .labeled_losses import ContrastiveLoss, MultiSimilarityLoss, SupervisedContrastiveLoss, TripletLoss
from .paired_loss import PairedEmbeddingLoss
from .paired_losses import (
    BarlowTwinsLoss,
    InfoNCELoss,
    NegativeCosineSimilarityLoss,
    NTXentLoss,
    SigLIPLoss,
    VICRegLoss,
)
from .proxy_loss import ProxyEmbeddingLoss
from .proxy_losses import ArcFaceLoss, CosFaceLoss, ProxyAnchorLoss

__all__ = [
    "PairedEmbeddingLoss",
    "BarlowTwinsLoss",
    "InfoNCELoss",
    "NTXentLoss",
    "NegativeCosineSimilarityLoss",
    "SigLIPLoss",
    "VICRegLoss",
    "LabeledEmbeddingLoss",
    "ContrastiveLoss",
    "MultiSimilarityLoss",
    "SupervisedContrastiveLoss",
    "TripletLoss",
    "ProxyEmbeddingLoss",
    "ArcFaceLoss",
    "CosFaceLoss",
    "ProxyAnchorLoss",
]
