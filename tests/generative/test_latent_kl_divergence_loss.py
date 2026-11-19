import pytest
import torch
from torch import distributions

from deep_learning_loss import LatentKLDivergenceLoss

MEANS: torch.Tensor = torch.tensor([[0.5, -1.0, 0.0], [1.5, 0.2, -0.3]])
LOG_VARIANCES: torch.Tensor = torch.tensor([[0.0, -1.0, 0.4], [0.3, -0.2, 0.0]])


def test_divergence_matches_torch_distributions() -> None:
    posterior = distributions.Normal(MEANS, (0.5 * LOG_VARIANCES).exp())
    prior = distributions.Normal(torch.zeros_like(MEANS), torch.ones_like(MEANS))
    expected = distributions.kl_divergence(posterior, prior).sum(dim=1).mean()
    assert LatentKLDivergenceLoss().compute(MEANS, LOG_VARIANCES).item() == pytest.approx(expected.item(), rel=1e-5)


def test_free_bits_floor_each_dimension() -> None:
    means = torch.zeros(4, 2, requires_grad=True)
    loss = LatentKLDivergenceLoss(free_bits=0.5).compute(means, torch.zeros(4, 2))
    (gradient,) = torch.autograd.grad(loss, means)
    assert loss.item() == pytest.approx(1.0)
    assert torch.equal(gradient, torch.zeros(4, 2))


def test_latent_divergence_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="log_variances"):
        LatentKLDivergenceLoss().compute(MEANS, LOG_VARIANCES[:1])
    with pytest.raises(ValueError, match="batch, \\*latent"):
        LatentKLDivergenceLoss().compute(MEANS[0], LOG_VARIANCES[0])
    with pytest.raises(ValueError, match="free_bits"):
        LatentKLDivergenceLoss(free_bits=-1.0)
