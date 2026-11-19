import pytest
import torch

from deep_learning_loss import DiceLoss, FocalTverskyLoss, JaccardLoss, LovaszSoftmaxLoss, SegmentationLoss, TverskyLoss

INDICES: torch.Tensor = torch.tensor([[[0, 1], [2, 2]]])
CONFIDENT_LOGITS: torch.Tensor = torch.nn.functional.one_hot(INDICES, 3).movedim(-1, 1).float() * 50.0


@pytest.mark.parametrize("loss", [DiceLoss(), JaccardLoss(), TverskyLoss(), FocalTverskyLoss(), LovaszSoftmaxLoss()])
def test_segmentation_losses_vanish_for_perfect_predictions(loss: SegmentationLoss) -> None:
    assert loss.compute(CONFIDENT_LOGITS, INDICES).item() == pytest.approx(0.0, abs=1e-3)


def test_dice_counts_overlap_per_class() -> None:
    # Class 1 is predicted on both top pixels but true on one; every other pixel is exact.
    logits = CONFIDENT_LOGITS.clone()
    logits[0, :, 0, 0] = torch.tensor([0.0, 50.0, 0.0])
    loss = DiceLoss(smoothing=0.0).compute(logits, INDICES)
    expected = ((1.0 - 0.0 / 1.0) + (1.0 - 2.0 / 3.0) + 0.0) / 3.0
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_balanced_tversky_is_dice() -> None:
    logits = torch.randn(2, 3, 4, 4)
    indices = torch.randint(0, 3, (2, 4, 4))
    tversky = TverskyLoss(smoothing=0.0, alpha=0.5, beta=0.5).compute(logits, indices)
    assert tversky.item() == pytest.approx(DiceLoss(smoothing=0.0).compute(logits, indices).item(), rel=1e-5)
    focal = FocalTverskyLoss(smoothing=0.0, gamma=1.0).compute(logits, indices)
    assert focal.item() == pytest.approx(TverskyLoss(smoothing=0.0).compute(logits, indices).item(), rel=1e-5)


def test_single_channel_is_binary_foreground() -> None:
    indices = torch.tensor([[[1, 0], [0, 0]]])
    logits = torch.full((1, 1, 2, 2), -50.0)
    logits[0, 0, 0, 0] = 50.0
    assert DiceLoss().compute(logits, indices).item() == pytest.approx(0.0, abs=1e-4)


def test_masked_pixels_do_not_count() -> None:
    indices = torch.tensor([[[0, 255], [1, 1]]])
    logits = torch.nn.functional.one_hot(torch.tensor([[[0, 0], [1, 1]]]), 2).movedim(-1, 1).float() * 50.0
    is_included = indices != 255
    assert DiceLoss().compute(logits, indices, is_included).item() == pytest.approx(0.0, abs=1e-4)


def _reference_lovasz_softmax(probabilities: torch.Tensor, indices: torch.Tensor) -> float:
    # One class at a time, following the reference implementation of Berman et al.
    flat_probabilities = probabilities.movedim(1, -1).reshape(-1, probabilities.shape[1])
    flat_indices = indices.flatten()
    losses: list[float] = []
    for index in range(probabilities.shape[1]):
        truths = (flat_indices == index).float()
        if truths.sum() == 0:
            continue
        errors = (truths - flat_probabilities[:, index]).abs()
        sorted_errors, order = errors.sort(descending=True)
        sorted_truths = truths[order]
        intersections = sorted_truths.sum() - sorted_truths.cumsum(0)
        unions = sorted_truths.sum() + (1.0 - sorted_truths).cumsum(0)
        jaccard = 1.0 - intersections / unions
        jaccard[1:] = jaccard[1:] - jaccard[:-1].clone()
        losses.append(float(torch.dot(sorted_errors, jaccard)))
    return sum(losses) / len(losses)


def test_lovasz_softmax_matches_reference() -> None:
    logits = torch.randn(2, 4, 5, 6)
    indices = torch.randint(0, 3, (2, 5, 6))
    expected = _reference_lovasz_softmax(torch.softmax(logits, dim=1), indices)
    assert LovaszSoftmaxLoss().compute(logits, indices).item() == pytest.approx(expected, rel=1e-5)


def test_lovasz_softmax_ignores_masked_pixels() -> None:
    logits = torch.randn(2, 3, 4, 4)
    indices = torch.randint(0, 3, (2, 4, 4))
    is_included = torch.zeros(2, 4, 4, dtype=torch.bool)
    is_included[0] = True
    expected = LovaszSoftmaxLoss().compute(logits[:1], indices[:1])
    masked = LovaszSoftmaxLoss().compute(logits, torch.where(is_included, indices, 255), is_included)
    assert masked.item() == pytest.approx(expected.item(), rel=1e-5)


def test_segmentation_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="batch, classes"):
        DiceLoss().compute(CONFIDENT_LOGITS, INDICES[:, :1])
    with pytest.raises(ValueError, match="smoothing"):
        DiceLoss(smoothing=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        TverskyLoss(alpha=0.0, beta=0.0)
    with pytest.raises(ValueError, match="gamma"):
        FocalTverskyLoss(gamma=0.0)
