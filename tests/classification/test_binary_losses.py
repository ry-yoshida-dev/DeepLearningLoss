import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import AsymmetricLoss, BinaryCrossEntropyLoss, BinaryFocalLoss

LOGITS: torch.Tensor = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.0, 4.0]])
TARGETS: torch.Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.3]])


def test_binary_cross_entropy_matches_torch_with_positive_weight() -> None:
    expected = functional.binary_cross_entropy_with_logits(LOGITS, TARGETS, pos_weight=torch.tensor(3.0))
    assert BinaryCrossEntropyLoss(positive_weight=3.0).compute(LOGITS, TARGETS).item() == pytest.approx(expected.item())


def test_binary_focal_without_focusing_is_cross_entropy() -> None:
    focal = BinaryFocalLoss(gamma=0.0, alpha=None).compute_elementwise(LOGITS, TARGETS)
    assert torch.allclose(focal, BinaryCrossEntropyLoss().compute_elementwise(LOGITS, TARGETS))


def test_binary_focal_down_weights_confident_predictions() -> None:
    logits = torch.tensor([4.0, 0.1])
    targets = torch.ones(2)
    ratios = BinaryFocalLoss(alpha=None).compute_elementwise(
        logits, targets
    ) / functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert ratios[0] < ratios[1] < 1.0


def test_asymmetric_without_focusing_or_clip_is_cross_entropy() -> None:
    asymmetric = AsymmetricLoss(gamma_positive=0.0, gamma_negative=0.0, clip=0.0).compute_elementwise(LOGITS, TARGETS)
    assert torch.allclose(asymmetric, BinaryCrossEntropyLoss().compute_elementwise(LOGITS, TARGETS), atol=1e-6)


def test_asymmetric_discards_easy_negatives() -> None:
    losses = AsymmetricLoss(clip=0.05).compute_elementwise(torch.tensor([-4.0]), torch.tensor([0.0]))
    assert losses.item() == 0.0


def test_binary_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="positive_weight"):
        BinaryCrossEntropyLoss(positive_weight=0.0)
    with pytest.raises(ValueError, match="alpha"):
        BinaryFocalLoss(alpha=1.5)
    with pytest.raises(ValueError, match="clip"):
        AsymmetricLoss(clip=1.0)
