from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor

from ..loss import Loss


@dataclass(frozen=True)
class CTCLoss(Loss):
    """
    Alignment-free sequence loss; trains speech and handwriting recognition without frame-level labels.

    Connectionist temporal classification loss (Graves et al., 2006): the negative log-probability of the target
    label sequence summed over every alignment of it to the output frames, with a blank class between labels, so the
    network learns from transcripts alone. Each sequence's loss is divided by its target length, then averaged.

    Attributes
    ----------
    blank : int
        Index of the blank class among the output classes.
    is_infinity_zeroed : bool
        Whether to zero the loss of sequences that no alignment can produce, such as targets longer than the input;
        otherwise they give an infinite loss.

    Raises
    ------
    ValueError
        If blank is negative.
    """

    blank: int = 0
    is_infinity_zeroed: bool = True

    def __post_init__(self) -> None:
        if self.blank < 0:
            raise ValueError(f"blank must be non-negative, got {self.blank}")

    def compute(self, logits: Tensor, targets: Tensor, logit_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        """
        Compute the mean loss over the sequences.

        Parameters
        ----------
        logits : Tensor
            Unnormalized class scores of each frame with shape ``(batch, frames, classes)``, the blank among the
            classes.
        targets : Tensor
            Integer label sequences padded to shape ``(batch, labels)``; they must not contain the blank.
        logit_lengths : Tensor
            Integer number of valid frames of each sequence, with shape ``(batch,)``.
        target_lengths : Tensor
            Integer number of valid labels of each sequence, with shape ``(batch,)``.

        Returns
        -------
        Tensor
            The scalar loss.

        Raises
        ------
        TypeError
            If the targets or lengths are not integers.
        ValueError
            If the shapes do not match or blank is not a class.
        """
        self._require_indices("targets", targets)
        self._require_indices("logit_lengths", logit_lengths)
        self._require_indices("target_lengths", target_lengths)
        if logits.ndim != 3:
            raise ValueError(f"logits must have shape (batch, frames, classes), got {tuple(logits.shape)}")
        batch_size: int = logits.shape[0]
        if targets.ndim != 2 or targets.shape[0] != batch_size:
            raise ValueError(f"targets shape {tuple(targets.shape)} must be ({batch_size}, labels)")
        if logit_lengths.shape != (batch_size,) or target_lengths.shape != (batch_size,):
            raise ValueError(
                f"logit_lengths shape {tuple(logit_lengths.shape)} and target_lengths shape "
                + f"{tuple(target_lengths.shape)} must be ({batch_size},)"
            )
        if self.blank >= logits.shape[2]:
            raise ValueError(f"blank {self.blank} must be one of the {logits.shape[2]} classes")
        # PyTorch expects the frames first.
        log_probabilities: Tensor = functional.log_softmax(logits, dim=2).transpose(0, 1)
        return functional.ctc_loss(
            log_probabilities,
            targets.long(),
            logit_lengths.long(),
            target_lengths.long(),
            blank=self.blank,
            reduction="mean",
            zero_infinity=self.is_infinity_zeroed,
        )
