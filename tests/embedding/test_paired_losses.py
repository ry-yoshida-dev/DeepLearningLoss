import math

import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import (
    BarlowTwinsLoss,
    InfoNCELoss,
    NegativeCosineSimilarityLoss,
    NTXentLoss,
    PairedEmbeddingLoss,
    SigLIPLoss,
    VICRegLoss,
)

GENERATOR: torch.Generator = torch.Generator().manual_seed(0)
FIRST: torch.Tensor = torch.randn(8, 16, generator=GENERATOR)
SECOND: torch.Tensor = FIRST + 0.1 * torch.randn(8, 16, generator=GENERATOR)
UNRELATED: torch.Tensor = torch.randn(8, 16, generator=GENERATOR)


@pytest.mark.parametrize(
    "loss", [InfoNCELoss(), NTXentLoss(), BarlowTwinsLoss(), VICRegLoss(), NegativeCosineSimilarityLoss()]
)
def test_paired_losses_prefer_matching_views(loss: PairedEmbeddingLoss) -> None:
    assert loss.compute(FIRST, SECOND).item() < loss.compute(FIRST, UNRELATED).item()


def test_info_nce_matches_cross_entropy_of_cosine_logits() -> None:
    logits = functional.normalize(FIRST, dim=1) @ functional.normalize(SECOND, dim=1).T / 0.1
    expected = functional.cross_entropy(logits, torch.arange(8))
    loss = InfoNCELoss(temperature=0.1, is_symmetric=False).compute(FIRST, SECOND)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_nt_xent_with_identical_orthogonal_views() -> None:
    views = torch.eye(4)
    # Each view matches its partner (cosine 1) and is orthogonal to the other 6 candidates (cosine 0).
    expected = -math.log(math.exp(2.0) / (math.exp(2.0) + 6.0))
    assert NTXentLoss(temperature=0.5).compute(views, views).item() == pytest.approx(expected, rel=1e-5)


def test_negative_cosine_similarity_is_minus_one_for_parallel_views() -> None:
    assert NegativeCosineSimilarityLoss().compute(FIRST[:1], 2.0 * FIRST[:1]).item() == pytest.approx(-1.0)


def test_paired_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="second"):
        InfoNCELoss().compute(FIRST, SECOND[:4])
    with pytest.raises(ValueError, match="batch, features"):
        InfoNCELoss().compute(FIRST[0], SECOND[0])
    with pytest.raises(ValueError, match="at least 2 pairs"):
        BarlowTwinsLoss().compute(FIRST[:1], SECOND[:1])
    with pytest.raises(ValueError, match="temperature"):
        NTXentLoss(temperature=0.0)
    with pytest.raises(ValueError, match="weights"):
        VICRegLoss(covariance_weight=-1.0)


def test_sig_lip_sums_logistic_losses_over_all_pairs() -> None:
    logits = functional.normalize(FIRST, dim=1) @ functional.normalize(SECOND, dim=1).T * 10.0 - 10.0
    labels = torch.eye(8)
    expected = functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / 8.0
    assert SigLIPLoss().compute(FIRST, SECOND).item() == pytest.approx(expected.item(), rel=1e-5)
    assert SigLIPLoss().compute(FIRST, SECOND).item() < SigLIPLoss().compute(FIRST, UNRELATED).item()
    with pytest.raises(ValueError, match="scale"):
        SigLIPLoss(scale=0.0)
