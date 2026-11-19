import math

import pytest
import torch

from deep_learning_loss import (
    AdversarialLoss,
    HingeAdversarialLoss,
    LeastSquaresAdversarialLoss,
    NonSaturatingAdversarialLoss,
    WassersteinAdversarialLoss,
)

REAL: torch.Tensor = torch.tensor([2.0, 0.0])
FAKE: torch.Tensor = torch.tensor([-1.0, 1.0])


@pytest.mark.parametrize(
    ("loss", "discriminator", "generator"),
    [
        (
            NonSaturatingAdversarialLoss(),
            (math.log1p(math.exp(-2.0)) + math.log(2.0)) / 2.0
            + (math.log1p(math.exp(-1.0)) + math.log1p(math.e)) / 2.0,
            (math.log1p(math.e) + math.log1p(math.exp(-1.0))) / 2.0,
        ),
        (LeastSquaresAdversarialLoss(), 0.5 * ((1.0 + 1.0) / 2.0 + (1.0 + 1.0) / 2.0), 0.5 * (4.0 + 0.0) / 2.0),
        (HingeAdversarialLoss(), (0.0 + 1.0) / 2.0 + (0.0 + 2.0) / 2.0, 0.0),
        (WassersteinAdversarialLoss(), 0.0 - 1.0, 0.0),
    ],
)
def test_adversarial_losses_compute_both_players(loss: AdversarialLoss, discriminator: float, generator: float) -> None:
    assert loss.compute_discriminator(REAL, FAKE).item() == pytest.approx(discriminator)
    assert loss.compute_generator(FAKE).item() == pytest.approx(generator)
