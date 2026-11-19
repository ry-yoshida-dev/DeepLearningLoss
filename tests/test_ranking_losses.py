import math

import pytest
import torch

from deep_learning_loss import ApproxNDCGLoss, LambdaRankLoss, ListMLELoss, ListNetLoss, RankingLoss, RankNetLoss

RELEVANCE: torch.Tensor = torch.tensor([[3.0, 2.0, 0.0, 1.0]])
ALIGNED: torch.Tensor = torch.tensor([[4.0, 3.0, 1.0, 2.0]])
REVERSED: torch.Tensor = -ALIGNED


ALL_LOSSES: list[RankingLoss] = [RankNetLoss(), ListNetLoss(), ListMLELoss(), ApproxNDCGLoss(), LambdaRankLoss()]


@pytest.mark.parametrize("loss", ALL_LOSSES)
def test_ranking_losses_prefer_the_relevance_order(loss: RankingLoss) -> None:
    assert loss.compute(ALIGNED, RELEVANCE).item() < loss.compute(REVERSED, RELEVANCE).item()


def test_rank_net_counts_only_ordered_pairs() -> None:
    loss = RankNetLoss().compute(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_list_mle_matches_plackett_luce() -> None:
    scores = torch.tensor([[1.0, 2.0]])
    # The first item is placed first among both, then the second alone with probability 1.
    expected = -math.log(math.e / (math.e + math.e**2))
    assert ListMLELoss().compute(scores, torch.tensor([[1.0, 0.0]])).item() == pytest.approx(expected)


def test_approx_ndcg_nears_zero_for_a_sharp_perfect_order() -> None:
    assert ApproxNDCGLoss(temperature=0.01).compute(ALIGNED, RELEVANCE).item() == pytest.approx(0.0, abs=1e-4)


def test_approx_ndcg_skips_lists_without_relevant_items() -> None:
    scores = torch.cat([ALIGNED, ALIGNED])
    relevance = torch.cat([RELEVANCE, torch.zeros_like(RELEVANCE)])
    expected = ApproxNDCGLoss().compute(ALIGNED, RELEVANCE)
    assert ApproxNDCGLoss().compute(scores, relevance).item() == pytest.approx(expected.item())


def test_ranking_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="relevance"):
        ListNetLoss().compute(ALIGNED, RELEVANCE[:, :2])
    with pytest.raises(ValueError, match="lists, items"):
        ListNetLoss().compute(ALIGNED[0], RELEVANCE[0])
    with pytest.raises(ValueError, match="temperature"):
        ApproxNDCGLoss(temperature=0.0)


@pytest.mark.parametrize("loss", ALL_LOSSES)
def test_padding_does_not_change_the_loss(loss: RankingLoss) -> None:
    scores = torch.tensor([[0.3, -1.0, 2.0, 0.5, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    relevance = torch.tensor([[3.0, 2.0, 0.0, 1.0, float("nan"), 9.0], [0.0, 2.0, 5.0, 5.0, 5.0, 5.0]])
    is_included = torch.tensor([[True] * 4 + [False] * 2, [True] * 2 + [False] * 4])
    padded = loss.compute(scores, relevance, is_included)
    separate = (loss.compute(scores[:1, :4], relevance[:1, :4]) + loss.compute(scores[1:, :2], relevance[1:, :2])) / 2.0
    assert padded.item() == pytest.approx(separate.item(), rel=1e-5)
    (gradient,) = torch.autograd.grad(padded, scores)
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~is_included], torch.zeros(6))


def test_lambda_rank_weights_pairs_by_ndcg_change() -> None:
    # Two items: the swap moves the gain 1 from rank 2 to rank 1, the full NDCG gap.
    scores = torch.tensor([[0.0, 0.0]])
    relevance = torch.tensor([[0.0, 1.0]])
    ideal = 1.0
    change = abs(1.0 - 1.0 / math.log2(3.0)) / ideal
    assert LambdaRankLoss().compute(scores, relevance).item() == pytest.approx(change * math.log(2.0), rel=1e-5)


def test_ranking_mask_is_validated() -> None:
    with pytest.raises(TypeError, match="boolean"):
        ListNetLoss().compute(ALIGNED, RELEVANCE, torch.ones_like(ALIGNED))
    with pytest.raises(ValueError, match="is_included"):
        ListNetLoss().compute(ALIGNED, RELEVANCE, torch.ones(1, 2, dtype=torch.bool))
