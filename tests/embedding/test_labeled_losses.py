import math

import pytest
import torch

from deep_learning_loss import (
    ContrastiveLoss,
    LabeledEmbeddingLoss,
    MultiSimilarityLoss,
    SupervisedContrastiveLoss,
    TripletLoss,
)

LABELS: torch.Tensor = torch.tensor([0, 0, 1, 1])
SEPARATED: torch.Tensor = torch.tensor([[0.0, 5.0], [0.0, 5.1], [5.0, 0.0], [5.1, 0.0]])
MIXED: torch.Tensor = torch.tensor([[0.0, 5.0], [5.0, 0.0], [0.0, 5.1], [5.1, 0.0]])
LOSSES: list[LabeledEmbeddingLoss] = [
    ContrastiveLoss(),
    TripletLoss(),
    SupervisedContrastiveLoss(),
    MultiSimilarityLoss(),
]


@pytest.mark.parametrize("loss", LOSSES)
def test_labeled_losses_prefer_clustered_labels(loss: LabeledEmbeddingLoss) -> None:
    assert loss.compute(SEPARATED, LABELS).item() < loss.compute(MIXED, LABELS).item()


def test_triplet_and_contrastive_vanish_beyond_the_margin() -> None:
    assert TripletLoss(margin=1.0).compute(SEPARATED, LABELS).item() == 0.0
    contrastive = ContrastiveLoss(margin=1.0).compute(SEPARATED, LABELS)
    assert contrastive.item() == pytest.approx(2.0 * 0.1**2 / 6.0, rel=1e-3)


@pytest.mark.parametrize("loss", LOSSES)
def test_labeled_losses_are_zero_without_usable_pairs(loss: LabeledEmbeddingLoss) -> None:
    embeddings = torch.randn(3, 2, requires_grad=True)
    value = loss.compute(
        embeddings, torch.tensor([0, 1, 2]) if not isinstance(loss, ContrastiveLoss) else torch.tensor([0, 0, 0])
    )
    value.backward()
    assert torch.isfinite(value)
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    single = torch.randn(1, 2, requires_grad=True)
    loss.compute(single, torch.tensor([0])).backward()
    assert single.grad is not None and torch.isfinite(single.grad).all()


def test_labeled_losses_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        TripletLoss().compute(SEPARATED, LABELS.float())
    with pytest.raises(ValueError, match="batch, features"):
        TripletLoss().compute(SEPARATED, LABELS[:2])
    with pytest.raises(ValueError, match="margin"):
        ContrastiveLoss(margin=0.0)
    with pytest.raises(ValueError, match="temperature"):
        SupervisedContrastiveLoss(temperature=0.0)


def test_multi_similarity_matches_formula_for_mined_pairs() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.8, 0.6]])
    labels = torch.tensor([0, 0, 1])
    loss = MultiSimilarityLoss(positive_scale=2.0, negative_scale=50.0, threshold=0.5, mining_margin=0.1)
    # Anchor 0: positive similarity 0.6 is below the negative 0.8 + 0.1, so both pairs are mined.
    first = math.log1p(math.exp(-2.0 * (0.6 - 0.5))) / 2.0 + math.log1p(math.exp(50.0 * (0.8 - 0.5))) / 50.0
    # Anchor 1: positive 0.6 is below the negative 0.96 + 0.1; anchor 2 has no positive.
    second = math.log1p(math.exp(-2.0 * (0.6 - 0.5))) / 2.0 + math.log1p(math.exp(50.0 * (0.96 - 0.5))) / 50.0
    assert loss.compute(embeddings, labels).item() == pytest.approx((first + second) / 2.0, rel=1e-4)


def test_multi_similarity_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="scales"):
        MultiSimilarityLoss(positive_scale=0.0)
    with pytest.raises(ValueError, match="mining_margin"):
        MultiSimilarityLoss(mining_margin=-0.1)
