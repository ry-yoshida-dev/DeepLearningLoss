import pytest
import torch

from deep_learning_loss import GradientDifferenceLoss, ImageLoss, SSIMLoss

GENERATOR: torch.Generator = torch.Generator().manual_seed(0)
IMAGES: torch.Tensor = torch.rand(2, 3, 16, 16, generator=GENERATOR)
NOISY: torch.Tensor = (IMAGES + 0.2 * torch.randn(2, 3, 16, 16, generator=GENERATOR)).clamp(0.0, 1.0)


@pytest.mark.parametrize("loss", [SSIMLoss(), GradientDifferenceLoss()])
def test_image_losses_vanish_for_identical_images_and_grow_with_noise(loss: ImageLoss) -> None:
    assert loss.compute(IMAGES, IMAGES).item() == pytest.approx(0.0, abs=1e-5)
    assert loss.compute(NOISY, IMAGES).item() > 0.01


def test_ssim_of_constant_images_is_the_luminance_term() -> None:
    # Double precision, so that the local variances are exactly zero.
    first = torch.full((1, 1, 11, 11), 0.2, dtype=torch.float64)
    second = torch.full((1, 1, 11, 11), 0.6, dtype=torch.float64)
    stability = 0.01**2
    expected = 1.0 - (2.0 * 0.2 * 0.6 + stability) / (0.2**2 + 0.6**2 + stability)
    assert SSIMLoss().compute(first, second).item() == pytest.approx(expected, rel=1e-4)


def test_gradient_difference_compares_edge_magnitudes() -> None:
    edge = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    # Horizontal gaps are 1 per row; inverting the edge keeps every magnitude.
    assert GradientDifferenceLoss().compute(torch.zeros_like(edge), edge).item() == pytest.approx(1.0)
    assert GradientDifferenceLoss().compute(1.0 - edge, edge).item() == pytest.approx(0.0)


def test_image_losses_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="batch, channels, height, width"):
        SSIMLoss().compute(IMAGES[0], IMAGES[0])
    with pytest.raises(ValueError, match="at least 11 x 11"):
        SSIMLoss().compute(IMAGES[..., :8], IMAGES[..., :8])
    with pytest.raises(ValueError, match="targets"):
        GradientDifferenceLoss().compute(IMAGES, IMAGES[:1])
    with pytest.raises(ValueError, match="window_size"):
        SSIMLoss(window_size=4)
    with pytest.raises(ValueError, match="data_range"):
        SSIMLoss(data_range=0.0)
    with pytest.raises(ValueError, match="exponent"):
        GradientDifferenceLoss(exponent=0.0)
