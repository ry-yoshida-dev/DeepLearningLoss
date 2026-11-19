import math

import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import (
    DistillationLoss,
    JensenShannonDivergenceLoss,
    KLDivergenceLoss,
    ReverseKLDivergenceLoss,
)

STUDENT: torch.Tensor = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 2.0]])
TEACHER: torch.Tensor = torch.tensor([[3.0, -1.0, 0.0], [0.0, 1.0, 1.0]])


@pytest.mark.parametrize("loss", [KLDivergenceLoss(), ReverseKLDivergenceLoss(), JensenShannonDivergenceLoss()])
def test_divergences_vanish_for_identical_logits(loss: DistillationLoss) -> None:
    assert loss.compute(STUDENT, STUDENT).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_matches_torch_with_temperature() -> None:
    temperature = 4.0
    expected = (
        functional.kl_div(
            functional.log_softmax(STUDENT / temperature, dim=1),
            functional.log_softmax(TEACHER / temperature, dim=1),
            reduction="batchmean",
            log_target=True,
        )
        * temperature**2
    )
    assert KLDivergenceLoss(temperature).compute(STUDENT, TEACHER).item() == pytest.approx(expected.item())


def test_reverse_kl_swaps_the_arguments() -> None:
    reverse = ReverseKLDivergenceLoss().compute(STUDENT, TEACHER)
    assert reverse.item() == pytest.approx(KLDivergenceLoss().compute(TEACHER, STUDENT).item())


def test_jensen_shannon_is_symmetric_and_bounded() -> None:
    loss = JensenShannonDivergenceLoss()
    assert loss.compute(STUDENT, TEACHER).item() == pytest.approx(loss.compute(TEACHER, STUDENT).item())
    opposite = loss.compute(torch.tensor([[100.0, -100.0]]), torch.tensor([[-100.0, 100.0]]))
    assert opposite.item() == pytest.approx(math.log(2.0), rel=1e-4)


def test_distillation_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="teacher_logits"):
        KLDivergenceLoss().compute(STUDENT, TEACHER[:1])
    with pytest.raises(ValueError, match="temperature"):
        KLDivergenceLoss(temperature=0.0)
