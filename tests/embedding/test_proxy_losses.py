import math

import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import ArcFaceLoss, CosFaceLoss, ProxyAnchorLoss, ProxyEmbeddingLoss

GENERATOR: torch.Generator = torch.Generator().manual_seed(0)
PROXIES: torch.Tensor = torch.randn(5, 8, generator=GENERATOR)
LABELS: torch.Tensor = torch.tensor([0, 1, 2, 2, 4, 0])
ALIGNED: torch.Tensor = PROXIES[LABELS] + 0.05 * torch.randn(6, 8, generator=GENERATOR)
SHUFFLED: torch.Tensor = PROXIES[(LABELS + 1) % 5]
COSINES: torch.Tensor = functional.normalize(ALIGNED, dim=1) @ functional.normalize(PROXIES, dim=1).T


@pytest.mark.parametrize("loss", [ArcFaceLoss(), CosFaceLoss(), ProxyAnchorLoss()])
def test_proxy_losses_prefer_embeddings_near_their_proxy(loss: ProxyEmbeddingLoss) -> None:
    assert loss.compute(ALIGNED, LABELS, PROXIES).item() < loss.compute(SHUFFLED, LABELS, PROXIES).item()


def test_zero_margins_give_the_scaled_cosine_softmax() -> None:
    expected = functional.cross_entropy(COSINES * 30.0, LABELS)
    assert ArcFaceLoss(scale=30.0, margin=0.0).compute(ALIGNED, LABELS, PROXIES).item() == pytest.approx(
        expected.item(), rel=1e-4
    )
    assert CosFaceLoss(scale=30.0, margin=0.0).compute(ALIGNED, LABELS, PROXIES).item() == pytest.approx(
        expected.item(), rel=1e-5
    )


def test_margins_shift_the_true_class_logit() -> None:
    true_cosines = COSINES.gather(1, LABELS.unsqueeze(1))
    arc_cosines = torch.cos(torch.acos(true_cosines) + 0.3)
    arc_expected = functional.cross_entropy(COSINES.scatter(1, LABELS.unsqueeze(1), arc_cosines) * 16.0, LABELS)
    arc = ArcFaceLoss(scale=16.0, margin=0.3).compute(ALIGNED, LABELS, PROXIES)
    assert arc.item() == pytest.approx(arc_expected.item(), rel=1e-4)
    cos_expected = functional.cross_entropy(COSINES.scatter(1, LABELS.unsqueeze(1), true_cosines - 0.2) * 16.0, LABELS)
    assert CosFaceLoss(scale=16.0, margin=0.2).compute(ALIGNED, LABELS, PROXIES).item() == pytest.approx(
        cos_expected.item(), rel=1e-5
    )


def test_proxy_anchor_matches_formula() -> None:
    loss = ProxyAnchorLoss(margin=0.1, alpha=4.0).compute(ALIGNED, LABELS, PROXIES)
    positive_terms: list[float] = []
    negative_terms: list[float] = []
    for proxy in range(5):
        is_positive = LABELS == proxy
        negatives = torch.exp(4.0 * (COSINES[~is_positive, proxy] + 0.1)).sum()
        negative_terms.append(math.log1p(negatives.item()))
        if is_positive.any():
            positives = torch.exp(-4.0 * (COSINES[is_positive, proxy] - 0.1)).sum()
            positive_terms.append(math.log1p(positives.item()))
    expected = sum(positive_terms) / len(positive_terms) + sum(negative_terms) / 5.0
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_proxy_losses_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        CosFaceLoss().compute(ALIGNED, LABELS.float(), PROXIES)
    with pytest.raises(ValueError, match="batch, features"):
        CosFaceLoss().compute(ALIGNED, LABELS[:2], PROXIES)
    with pytest.raises(ValueError, match="classes, features"):
        CosFaceLoss().compute(ALIGNED, LABELS, PROXIES[:, :3])
    with pytest.raises(ValueError, match="margin"):
        ArcFaceLoss(margin=2.0)
    with pytest.raises(ValueError, match="scale"):
        CosFaceLoss(scale=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ProxyAnchorLoss(alpha=0.0)
