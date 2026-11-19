import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import (
    CrossEntropyLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    MulticlassHingeLoss,
    Poly1Loss,
)

LOGITS: torch.Tensor = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.2, 0.1], [-1.0, 3.0, 0.0]])
INDICES: torch.Tensor = torch.tensor([0, 2, 1])
WEIGHTS: torch.Tensor = torch.tensor([1.0, 2.0, 0.5])


def test_cross_entropy_matches_torch_with_class_weights() -> None:
    expected = functional.cross_entropy(LOGITS, INDICES, weight=WEIGHTS)
    assert CrossEntropyLoss().compute(LOGITS, INDICES, WEIGHTS).item() == pytest.approx(expected.item())


def test_cross_entropy_smooths_labels() -> None:
    expected = functional.cross_entropy(LOGITS, INDICES, label_smoothing=0.1)
    assert CrossEntropyLoss(label_smoothing=0.1).compute(LOGITS, INDICES).item() == pytest.approx(expected.item())


def test_neutral_settings_reduce_to_cross_entropy() -> None:
    cross_entropy = CrossEntropyLoss().compute_elementwise(LOGITS, INDICES)
    for loss in (FocalLoss(gamma=0.0), Poly1Loss(epsilon=0.0)):
        assert torch.allclose(loss.compute_elementwise(LOGITS, INDICES), cross_entropy)


def test_focal_loss_down_weights_confident_predictions() -> None:
    ratios = FocalLoss(gamma=2.0).compute_elementwise(LOGITS, INDICES) / CrossEntropyLoss().compute_elementwise(
        LOGITS, INDICES
    )
    assert torch.all(ratios < 1.0)
    assert ratios[1] > ratios[0] and ratios[1] > ratios[2]


def test_generalized_cross_entropy_with_unit_q_is_one_minus_probability() -> None:
    probabilities = functional.softmax(LOGITS, dim=1).gather(1, INDICES.unsqueeze(1)).squeeze(1)
    losses = GeneralizedCrossEntropyLoss(q=1.0).compute_elementwise(LOGITS, INDICES)
    assert torch.allclose(losses, 1.0 - probabilities)


def test_multiclass_hinge_matches_torch() -> None:
    expected = functional.multi_margin_loss(LOGITS, INDICES, margin=0.5)
    assert MulticlassHingeLoss(margin=0.5).compute(LOGITS, INDICES).item() == pytest.approx(expected.item())


def test_classification_losses_handle_positions_and_ignored_labels() -> None:
    logits = torch.randn(2, 3, 4, 5)
    indices = torch.randint(0, 3, (2, 4, 5))
    is_included = torch.rand(2, 4, 5) > 0.3
    ignored = torch.where(is_included, indices, 255)
    expected = functional.cross_entropy(logits, ignored, ignore_index=255)
    assert CrossEntropyLoss().compute(logits, ignored, is_included=is_included).item() == pytest.approx(expected.item())


def test_classification_losses_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        CrossEntropyLoss().compute(LOGITS, INDICES.float())
    with pytest.raises(ValueError, match="batch, classes"):
        CrossEntropyLoss().compute(LOGITS, INDICES[:2])
    with pytest.raises(ValueError, match="class_weights"):
        CrossEntropyLoss().compute(LOGITS, INDICES, WEIGHTS[:2])


def test_classification_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        CrossEntropyLoss(label_smoothing=1.0)
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        Poly1Loss(epsilon=-2.0)
    with pytest.raises(ValueError, match="q"):
        GeneralizedCrossEntropyLoss(q=0.0)
    with pytest.raises(ValueError, match="margin"):
        MulticlassHingeLoss(margin=0.0)
