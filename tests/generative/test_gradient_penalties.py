import pytest
import torch

from deep_learning_loss import GradientPenalty, R1GradientPenalty, WassersteinGradientPenalty

WEIGHTS: torch.Tensor = torch.tensor([3.0, 4.0])


def _discriminate(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return inputs @ weights


def test_penalties_match_the_squared_and_centered_norms() -> None:
    inputs = torch.randn(5, 2, requires_grad=True)
    scores = _discriminate(inputs, WEIGHTS)
    # The gradient is the weight vector, of norm 5, for every sample.
    assert R1GradientPenalty().compute(inputs, scores).item() == pytest.approx(25.0)
    scores = _discriminate(inputs, WEIGHTS)
    assert WassersteinGradientPenalty().compute(inputs, scores).item() == pytest.approx(16.0)


@pytest.mark.parametrize("penalty", [R1GradientPenalty(), WassersteinGradientPenalty()])
def test_penalties_train_the_discriminator(penalty: GradientPenalty) -> None:
    weights = WEIGHTS.clone().requires_grad_()
    inputs = torch.randn(5, 2, requires_grad=True)
    (gradient,) = torch.autograd.grad(penalty.compute(inputs, _discriminate(inputs, weights)), weights)
    assert gradient.abs().sum().item() > 0.0


def test_interpolate_mixes_each_pair_on_its_segment() -> None:
    real, fake = torch.zeros(64, 3), torch.ones(64, 3)
    mixed = WassersteinGradientPenalty.interpolate(real, fake)
    assert mixed.requires_grad and mixed.shape == real.shape
    assert ((mixed >= 0.0) & (mixed <= 1.0)).all()
    assert torch.equal(mixed[:, 0], mixed[:, 2])


def test_penalties_reject_invalid_inputs() -> None:
    inputs = torch.randn(5, 2)
    with pytest.raises(ValueError, match="require gradients"):
        R1GradientPenalty().compute(inputs, _discriminate(inputs, WEIGHTS))
    inputs.requires_grad_()
    with pytest.raises(ValueError, match="batch axis"):
        R1GradientPenalty().compute(inputs, _discriminate(inputs, WEIGHTS).sum())
    with pytest.raises(ValueError, match="must match"):
        WassersteinGradientPenalty.interpolate(torch.zeros(2, 3), torch.zeros(3, 3))
