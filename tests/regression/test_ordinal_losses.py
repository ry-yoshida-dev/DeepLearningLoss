import pytest
import torch
import torch.nn.functional as functional

from deep_learning_loss import CORALLoss, CORNLoss, OrdinalRegressionLoss

LOGITS: torch.Tensor = torch.tensor([[2.0, 0.5, -1.0], [-1.0, -2.0, -3.0]])
INDICES: torch.Tensor = torch.tensor([2, 0])


def test_coral_sums_binary_cross_entropies_over_thresholds() -> None:
    levels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = functional.binary_cross_entropy_with_logits(LOGITS, levels, reduction="none").sum(dim=1).mean()
    assert CORALLoss().compute(LOGITS, INDICES).item() == pytest.approx(expected.item())


def test_corn_trains_each_threshold_on_the_samples_above_the_previous_one() -> None:
    # Rank 2 conditions all three thresholds; rank 0 conditions only the first.
    first = functional.binary_cross_entropy_with_logits(LOGITS[0], torch.tensor([1.0, 1.0, 0.0]), reduction="sum")
    second = functional.binary_cross_entropy_with_logits(LOGITS[1, 0], torch.tensor(0.0))
    assert CORNLoss().compute(LOGITS, INDICES).item() == pytest.approx(((first + second) / 4.0).item())


def test_ranks_decode_the_exceedance_probabilities() -> None:
    assert torch.equal(CORALLoss().compute_ranks(LOGITS), torch.tensor([2, 0]))
    # Conditional probabilities 0.88, 0.62 multiply to 0.55, then 0.15.
    assert torch.equal(CORNLoss().compute_ranks(LOGITS), torch.tensor([2, 0]))


@pytest.mark.parametrize("loss", [CORALLoss(), CORNLoss()])
def test_ordinal_losses_accept_positions_and_prefer_the_true_rank(loss: OrdinalRegressionLoss) -> None:
    indices = torch.randint(0, 4, (2, 5))
    thresholds = torch.arange(3).view(1, 3, 1)
    confident = torch.where(indices.unsqueeze(1) > thresholds, 10.0, -10.0)
    assert loss.compute(confident, indices).item() < loss.compute(-confident, indices).item()
    assert torch.equal(loss.compute_ranks(confident), indices)


def test_ordinal_losses_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        CORALLoss().compute(LOGITS, INDICES.float())
    with pytest.raises(ValueError, match="ranks - 1"):
        CORALLoss().compute(LOGITS, INDICES[:1])
    with pytest.raises(ValueError, match="ranks - 1"):
        CORNLoss().compute_ranks(LOGITS[0, 0])
