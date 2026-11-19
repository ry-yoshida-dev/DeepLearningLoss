import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import (
    CrossEntropyLoss,
    FocalLoss,
    SoftLabelClassificationLoss,
    SoftLabelCrossEntropyLoss,
    SoftLabelFocalLoss,
)

LOGITS: torch.Tensor = torch.randn(4, 3, 5)
INDICES: torch.Tensor = torch.randint(0, 3, (4, 5))
ONE_HOT: torch.Tensor = functional.one_hot(INDICES, 3).movedim(-1, 1).float()


def test_one_hot_targets_give_the_hard_label_losses() -> None:
    soft = SoftLabelCrossEntropyLoss().compute_elementwise(LOGITS, ONE_HOT)
    assert torch.allclose(soft, CrossEntropyLoss().compute_elementwise(LOGITS, INDICES), atol=1e-6)
    focal = SoftLabelFocalLoss(gamma=2.0).compute_elementwise(LOGITS, ONE_HOT)
    assert torch.allclose(focal, FocalLoss(gamma=2.0).compute_elementwise(LOGITS, INDICES), atol=1e-6)


def test_mixed_targets_interpolate_the_losses() -> None:
    logits = torch.randn(6, 4)
    first, second = (
        functional.one_hot(torch.arange(6) % 4, 4).float(),
        functional.one_hot(torch.arange(6) // 2, 4).float(),
    )
    mixed = SoftLabelCrossEntropyLoss().compute(logits, 0.3 * first + 0.7 * second)
    expected = 0.3 * SoftLabelCrossEntropyLoss().compute(logits, first) + 0.7 * SoftLabelCrossEntropyLoss().compute(
        logits, second
    )
    assert mixed.item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.parametrize("loss", [SoftLabelCrossEntropyLoss(), SoftLabelFocalLoss()])
def test_masked_positions_do_not_count(loss: SoftLabelClassificationLoss) -> None:
    is_included = torch.rand(4, 5) > 0.3
    targets = torch.where(is_included.unsqueeze(1), ONE_HOT, float("nan"))
    expected = loss.compute_elementwise(LOGITS, ONE_HOT)[is_included].mean()
    assert loss.compute(LOGITS, targets, is_included).item() == pytest.approx(expected.item(), rel=1e-5)


def test_soft_label_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        SoftLabelCrossEntropyLoss().compute(LOGITS, ONE_HOT[:, :2])
    with pytest.raises(ValueError, match="batch, classes"):
        SoftLabelCrossEntropyLoss().compute(LOGITS[0, 0], ONE_HOT[0, 0])
    with pytest.raises(ValueError, match="batch, \\*positions"):
        SoftLabelCrossEntropyLoss().compute(LOGITS, ONE_HOT, torch.ones(4, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        SoftLabelCrossEntropyLoss().compute(LOGITS, ONE_HOT, torch.ones(4, 5))
    with pytest.raises(ValueError, match="gamma"):
        SoftLabelFocalLoss(gamma=-1.0)
