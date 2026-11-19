import math

import pytest
import torch

from deep_learning_loss import BoxRegressionLoss, CIoULoss, DIoULoss, GIoULoss, IoULoss

TARGETS: torch.Tensor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
SHIFTED: torch.Tensor = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
DISJOINT: torch.Tensor = torch.tensor([[3.0, 0.0, 5.0, 2.0]])


@pytest.mark.parametrize("loss", [IoULoss(), GIoULoss(), DIoULoss(), CIoULoss()])
def test_box_losses_vanish_for_identical_boxes(loss: BoxRegressionLoss) -> None:
    assert loss.compute(TARGETS, TARGETS).item() == pytest.approx(0.0, abs=1e-6)


def test_overlap_terms_match_hand_computation() -> None:
    # Intersection 1, union 7, enclosing box 3 x 3 with diagonal 18, centers 2 apart squared.
    iou = 1.0 / 7.0
    assert IoULoss().compute(SHIFTED, TARGETS).item() == pytest.approx(1.0 - iou, rel=1e-5)
    assert GIoULoss().compute(SHIFTED, TARGETS).item() == pytest.approx(1.0 - iou + 2.0 / 9.0, rel=1e-5)
    assert DIoULoss().compute(SHIFTED, TARGETS).item() == pytest.approx(1.0 - iou + 2.0 / 18.0, rel=1e-5)
    # Same aspect ratio, so CIoU reduces to DIoU.
    assert CIoULoss().compute(SHIFTED, TARGETS).item() == pytest.approx(1.0 - iou + 2.0 / 18.0, rel=1e-5)


def test_ciou_penalizes_aspect_ratio() -> None:
    wide = torch.tensor([[-1.0, 0.5, 3.0, 1.5]])
    gap = (math.atan(1.0) - math.atan(4.0)) ** 2 * 4.0 / math.pi**2
    difference = CIoULoss().compute(wide, TARGETS) - DIoULoss().compute(wide, TARGETS)
    iou = 2.0 / 6.0
    assert difference.item() == pytest.approx(gap / (1.0 - iou + gap) * gap, rel=1e-4)


def test_generalized_losses_give_gradient_without_overlap() -> None:
    assert IoULoss().compute(DISJOINT, TARGETS).item() == pytest.approx(1.0)
    for loss in [GIoULoss(), DIoULoss(), CIoULoss()]:
        predictions = DISJOINT.clone().requires_grad_()
        (gradient,) = torch.autograd.grad(loss.compute(predictions, TARGETS), predictions)
        assert gradient.abs().sum().item() > 0.0


def test_masks_select_boxes() -> None:
    predictions = torch.cat([SHIFTED, TARGETS])
    targets = torch.cat([TARGETS, torch.full((1, 4), float("nan"))])
    is_included = torch.tensor([True, False])
    predictions.requires_grad_()
    value = GIoULoss().compute_masked(predictions, targets, is_included)
    (gradient,) = torch.autograd.grad(value, predictions)
    assert value.item() == pytest.approx(GIoULoss().compute(SHIFTED, TARGETS).item())
    assert torch.isfinite(gradient).all()


def test_box_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="size 4"):
        IoULoss().compute(TARGETS[:, :3], TARGETS[:, :3])
    with pytest.raises(ValueError, match="targets"):
        IoULoss().compute(TARGETS, torch.cat([TARGETS, TARGETS]))
    with pytest.raises(ValueError, match="is_included"):
        IoULoss().compute_masked(TARGETS, TARGETS, torch.ones(1, 4, dtype=torch.bool))
