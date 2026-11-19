from __future__ import annotations

from enum import Enum

from .classification import (
    AsymmetricLoss,
    BinaryCrossEntropyLoss,
    BinaryFocalLoss,
    CrossEntropyLoss,
    CTCLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    MulticlassHingeLoss,
    Poly1Loss,
    SoftLabelCrossEntropyLoss,
    SoftLabelFocalLoss,
)
from .distillation_losses import JensenShannonDivergenceLoss, KLDivergenceLoss, ReverseKLDivergenceLoss
from .embedding import (
    ArcFaceLoss,
    BarlowTwinsLoss,
    ContrastiveLoss,
    CosFaceLoss,
    InfoNCELoss,
    MultiSimilarityLoss,
    NegativeCosineSimilarityLoss,
    NTXentLoss,
    ProxyAnchorLoss,
    SigLIPLoss,
    SupervisedContrastiveLoss,
    TripletLoss,
    VICRegLoss,
)
from .generative import (
    HingeAdversarialLoss,
    LatentKLDivergenceLoss,
    LeastSquaresAdversarialLoss,
    NonSaturatingAdversarialLoss,
    R1GradientPenalty,
    WassersteinAdversarialLoss,
    WassersteinGradientPenalty,
)
from .loss import Loss
from .ranking_losses import ApproxNDCGLoss, LambdaRankLoss, ListMLELoss, ListNetLoss, RankNetLoss
from .regression import (
    BetaNLLLoss,
    CharbonnierLoss,
    CORALLoss,
    CORNLoss,
    GammaNLLLoss,
    GaussianCRPSLoss,
    GaussianMixtureNLLLoss,
    GaussianNLLLoss,
    HuberLoss,
    LaplaceNLLLoss,
    LogCoshLoss,
    LogNormalNLLLoss,
    MAELoss,
    MAPELoss,
    MSELoss,
    MSLELoss,
    NegativeBinomialNLLLoss,
    PinballLoss,
    PoissonNLLLoss,
    QuantileLoss,
    SMAPELoss,
    StudentTNLLLoss,
    ZeroInflatedNegativeBinomialNLLLoss,
    ZeroInflatedPoissonNLLLoss,
)
from .vision import (
    CIoULoss,
    DiceLoss,
    DIoULoss,
    FocalTverskyLoss,
    GIoULoss,
    GradientDifferenceLoss,
    IoULoss,
    JaccardLoss,
    LovaszSoftmaxLoss,
    SSIMLoss,
    TverskyLoss,
)


class LearningTask(Enum):
    """
    Learning tasks, each mapped to the losses that suit it, the usual default first.

    Attributes
    ----------
    REGRESSION : Predict continuous values.
    QUANTILE_REGRESSION : Predict quantiles of continuous values, such as prediction intervals.
    PROBABILISTIC_REGRESSION : Predict a distribution over continuous values, expressing uncertainty.
    COUNT_REGRESSION : Predict non-negative counts.
    BOUNDED_REGRESSION : Predict positive values or proportions between 0 and 1.
    ORDINAL_REGRESSION : Predict one of several ordered classes, such as a rating or a severity level.
    RECONSTRUCTION : Reproduce the input, as in autoencoders, denoising, imputation and image restoration.
    BINARY_CLASSIFICATION : Choose between two classes.
    MULTICLASS_CLASSIFICATION : Choose one of several classes.
    MULTI_LABEL_CLASSIFICATION : Choose any number of several classes.
    SEMANTIC_SEGMENTATION : Classify each pixel.
    OBJECT_DETECTION : Locate objects with bounding boxes and classify them.
    SEQUENCE_TRANSCRIPTION : Transcribe frames into a shorter label sequence, as in speech or handwriting recognition.
    KNOWLEDGE_DISTILLATION : Train a student to mimic a teacher's class distribution.
    SELF_SUPERVISED_LEARNING : Learn representations from paired views without labels, or align two modalities.
    METRIC_LEARNING : Learn embeddings whose distances reflect labels, for retrieval and verification.
    LEARNING_TO_RANK : Order candidate items by relevance.
    VARIATIONAL_AUTOENCODING : Learn a latent distribution that reconstructs the input.
    ADVERSARIAL_GENERATION : Train a generator against a discriminator.
    """

    REGRESSION = "regression"
    QUANTILE_REGRESSION = "quantile_regression"
    PROBABILISTIC_REGRESSION = "probabilistic_regression"
    COUNT_REGRESSION = "count_regression"
    BOUNDED_REGRESSION = "bounded_regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    RECONSTRUCTION = "reconstruction"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OBJECT_DETECTION = "object_detection"
    SEQUENCE_TRANSCRIPTION = "sequence_transcription"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    SELF_SUPERVISED_LEARNING = "self_supervised_learning"
    METRIC_LEARNING = "metric_learning"
    LEARNING_TO_RANK = "learning_to_rank"
    VARIATIONAL_AUTOENCODING = "variational_autoencoding"
    ADVERSARIAL_GENERATION = "adversarial_generation"

    @property
    def losses(self) -> list[type[Loss]]:
        """
        Losses that suit the task, the usual default first.

        Returns
        -------
        list[type[Loss]]
            The loss classes; losses of other families, such as a ``DiceLoss`` next to a ``CrossEntropyLoss``, are
            often added together.
        """
        match self:
            case LearningTask.REGRESSION:
                return [MSELoss, MAELoss, HuberLoss, LogCoshLoss, MSLELoss, MAPELoss, SMAPELoss]
            case LearningTask.QUANTILE_REGRESSION:
                return [QuantileLoss, PinballLoss]
            case LearningTask.PROBABILISTIC_REGRESSION:
                return [
                    GaussianNLLLoss,
                    GaussianCRPSLoss,
                    LaplaceNLLLoss,
                    StudentTNLLLoss,
                    GaussianMixtureNLLLoss,
                    QuantileLoss,
                ]
            case LearningTask.COUNT_REGRESSION:
                return [
                    PoissonNLLLoss,
                    NegativeBinomialNLLLoss,
                    ZeroInflatedPoissonNLLLoss,
                    ZeroInflatedNegativeBinomialNLLLoss,
                    MSLELoss,
                ]
            case LearningTask.BOUNDED_REGRESSION:
                return [GammaNLLLoss, LogNormalNLLLoss, BetaNLLLoss]
            case LearningTask.ORDINAL_REGRESSION:
                return [CORALLoss, CORNLoss]
            case LearningTask.RECONSTRUCTION:
                return [MSELoss, MAELoss, CharbonnierLoss, HuberLoss, SSIMLoss, GradientDifferenceLoss]
            case LearningTask.BINARY_CLASSIFICATION:
                return [BinaryCrossEntropyLoss, BinaryFocalLoss]
            case LearningTask.MULTICLASS_CLASSIFICATION:
                return [
                    CrossEntropyLoss,
                    FocalLoss,
                    Poly1Loss,
                    GeneralizedCrossEntropyLoss,
                    MulticlassHingeLoss,
                    SoftLabelCrossEntropyLoss,
                    SoftLabelFocalLoss,
                ]
            case LearningTask.MULTI_LABEL_CLASSIFICATION:
                return [BinaryCrossEntropyLoss, AsymmetricLoss, BinaryFocalLoss]
            case LearningTask.SEMANTIC_SEGMENTATION:
                return [
                    CrossEntropyLoss,
                    DiceLoss,
                    FocalLoss,
                    JaccardLoss,
                    LovaszSoftmaxLoss,
                    TverskyLoss,
                    FocalTverskyLoss,
                ]
            case LearningTask.OBJECT_DETECTION:
                return [GIoULoss, CIoULoss, DIoULoss, IoULoss, BinaryFocalLoss, HuberLoss]
            case LearningTask.SEQUENCE_TRANSCRIPTION:
                return [CTCLoss]
            case LearningTask.KNOWLEDGE_DISTILLATION:
                return [KLDivergenceLoss, ReverseKLDivergenceLoss, JensenShannonDivergenceLoss]
            case LearningTask.SELF_SUPERVISED_LEARNING:
                return [InfoNCELoss, SigLIPLoss, NTXentLoss, BarlowTwinsLoss, VICRegLoss, NegativeCosineSimilarityLoss]
            case LearningTask.METRIC_LEARNING:
                return [
                    SupervisedContrastiveLoss,
                    ArcFaceLoss,
                    CosFaceLoss,
                    ProxyAnchorLoss,
                    MultiSimilarityLoss,
                    TripletLoss,
                    ContrastiveLoss,
                ]
            case LearningTask.LEARNING_TO_RANK:
                return [ListNetLoss, LambdaRankLoss, ApproxNDCGLoss, ListMLELoss, RankNetLoss]
            case LearningTask.VARIATIONAL_AUTOENCODING:
                return [LatentKLDivergenceLoss, MSELoss, BinaryCrossEntropyLoss, GaussianNLLLoss]
            case LearningTask.ADVERSARIAL_GENERATION:
                return [
                    NonSaturatingAdversarialLoss,
                    HingeAdversarialLoss,
                    LeastSquaresAdversarialLoss,
                    WassersteinAdversarialLoss,
                    R1GradientPenalty,
                    WassersteinGradientPenalty,
                ]

    def describe(self) -> str:
        """
        Describe the losses that suit the task, one per line, for choosing among them.

        Returns
        -------
        str
            The task followed by each loss name and summary, the usual default first.
        """
        lines: list[str] = [f"{self.name}:"] + [f"- {loss.__name__}: {loss.summarize()}" for loss in self.losses]
        return "\n".join(lines)

    @classmethod
    def find(cls, loss: type[Loss]) -> list[LearningTask]:
        """
        Find the tasks that a loss suits.

        Parameters
        ----------
        loss : type[Loss]
            The loss class.

        Returns
        -------
        list[LearningTask]
            The tasks listing the loss; empty for a loss defined outside this package.
        """
        return [task for task in cls if loss in task.losses]
