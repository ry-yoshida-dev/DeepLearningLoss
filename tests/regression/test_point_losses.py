import math

import pytest
import torch

from deep_learning_loss import (
    CharbonnierLoss,
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MAPELoss,
    MSELoss,
    MSLELoss,
    PinballLoss,
    PointLoss,
    SMAPELoss,
)

PREDICTIONS: torch.Tensor = torch.tensor([0.0, 1.0, -3.0])
TARGETS: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (MSELoss(), [1.0, 0.0, 16.0]),
        (MAELoss(), [1.0, 0.0, 4.0]),
        (HuberLoss(delta=2.0), [0.5, 0.0, 2.0 * (4.0 - 1.0)]),
        (LogCoshLoss(), [math.log(math.cosh(1.0)), 0.0, math.log(math.cosh(4.0))]),
        (PinballLoss(0.9), [0.9, 0.0, 3.6]),
        (CharbonnierLoss(epsilon=1e-3), [math.sqrt(1.0 + 1e-6), 1e-3, math.sqrt(16.0 + 1e-6)]),
        (MAPELoss(), [1.0, 0.0, 4.0]),
        (SMAPELoss(), [2.0, 0.0, 2.0 * 4.0 / 4.0]),
    ],
)
def test_point_losses_compute_each_value(loss: PointLoss, expected: list[float]) -> None:
    assert torch.allclose(loss.compute_elementwise(PREDICTIONS, TARGETS), torch.tensor(expected))
    assert loss.compute(PREDICTIONS, TARGETS).item() == pytest.approx(sum(expected) / 3.0, rel=1e-6)


def test_msle_compares_logarithms() -> None:
    value = MSLELoss().compute(torch.tensor([math.e - 1.0]), torch.tensor([0.0]))
    assert value.item() == pytest.approx(1.0)


def test_pinball_loss_penalizes_over_prediction_by_complement() -> None:
    assert PinballLoss(0.9).compute(torch.tensor([2.0]), torch.tensor([1.0])).item() == pytest.approx(0.1)


def test_log_cosh_loss_does_not_overflow() -> None:
    value = LogCoshLoss().compute(torch.tensor([1000.0]), torch.tensor([0.0]))
    assert value.item() == pytest.approx(1000.0 - math.log(2.0))


def test_masked_mean_averages_selected_values() -> None:
    is_included = torch.tensor([True, False, True])
    assert MAELoss().compute_masked(PREDICTIONS, TARGETS, is_included).item() == pytest.approx(2.5)
    assert MAELoss().compute_masked(PREDICTIONS, TARGETS, is_included & False).item() == 0.0


def test_masked_mean_ignores_nan_targets_in_gradients() -> None:
    predictions = torch.tensor([0.0, 1.0], requires_grad=True)
    targets = torch.tensor([1.0, float("nan")])
    loss = MSELoss().compute_masked(predictions, targets, torch.tensor([True, False]))
    loss.backward()
    assert loss.item() == pytest.approx(1.0)
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_point_losses_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(3, 1\)"):
        MSELoss().compute(PREDICTIONS.unsqueeze(1), TARGETS)
    with pytest.raises(ValueError, match="is_included"):
        MSELoss().compute_masked(PREDICTIONS, TARGETS, torch.tensor([True]))
    with pytest.raises(TypeError, match="boolean"):
        MSELoss().compute_masked(PREDICTIONS, TARGETS, torch.ones(3))


def test_point_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="delta"):
        HuberLoss(delta=0.0)
    with pytest.raises(ValueError, match="level"):
        PinballLoss(1.0)
    with pytest.raises(ValueError, match="epsilon"):
        CharbonnierLoss(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        MAPELoss(epsilon=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        SMAPELoss(epsilon=0.0)
