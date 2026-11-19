import math

import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import CTCLoss

LOGITS: torch.Tensor = torch.randn(2, 6, 4)
TARGETS: torch.Tensor = torch.tensor([[1, 2, 2], [3, 1, 0]])
LOGIT_LENGTHS: torch.Tensor = torch.tensor([6, 4])
TARGET_LENGTHS: torch.Tensor = torch.tensor([3, 2])


def test_ctc_matches_pytorch_with_batch_first_logits() -> None:
    expected = functional.ctc_loss(
        functional.log_softmax(LOGITS, dim=2).transpose(0, 1),
        TARGETS,
        LOGIT_LENGTHS,
        TARGET_LENGTHS,
        zero_infinity=True,
    )
    assert CTCLoss().compute(LOGITS, TARGETS, LOGIT_LENGTHS, TARGET_LENGTHS).item() == pytest.approx(expected.item())


def test_ctc_sums_over_alignments() -> None:
    # One frame per label with nothing to spare: the only alignment emits the labels in order.
    logits = torch.zeros(1, 2, 3)
    loss = CTCLoss().compute(logits, torch.tensor([[1, 2]]), torch.tensor([2]), torch.tensor([2]))
    assert loss.item() == pytest.approx(2.0 * math.log(3.0) / 2.0, rel=1e-5)


def test_impossible_alignments_are_zeroed_or_infinite() -> None:
    lengths = (torch.tensor([1]), torch.tensor([3]))
    targets = torch.tensor([[1, 2, 1]])
    assert CTCLoss().compute(LOGITS[:1], targets, *lengths).item() == 0.0
    assert math.isinf(CTCLoss(is_infinity_zeroed=False).compute(LOGITS[:1], targets, *lengths).item())


def test_ctc_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="targets"):
        CTCLoss().compute(LOGITS, TARGETS.float(), LOGIT_LENGTHS, TARGET_LENGTHS)
    with pytest.raises(ValueError, match="batch, frames, classes"):
        CTCLoss().compute(LOGITS[0], TARGETS, LOGIT_LENGTHS, TARGET_LENGTHS)
    with pytest.raises(ValueError, match="target_lengths"):
        CTCLoss().compute(LOGITS, TARGETS, LOGIT_LENGTHS, TARGET_LENGTHS[:1])
    with pytest.raises(ValueError, match="blank 4"):
        CTCLoss(blank=4).compute(LOGITS, TARGETS, LOGIT_LENGTHS, TARGET_LENGTHS)
    with pytest.raises(ValueError, match="blank"):
        CTCLoss(blank=-1)
