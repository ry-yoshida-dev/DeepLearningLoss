from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..image_loss import ImageLoss


@dataclass(frozen=True)
class SSIMLoss(ImageLoss):
    """
    Structural similarity loss; matches local luminance, contrast and structure as human vision perceives them.

    SSIM loss (Wang et al., 2004) ``1 - SSIM``: compares the local means, variances and covariance of the two images
    in a Gaussian window, so it keeps edges and textures that a pixel loss blurs; often added to an MAE loss for image
    restoration (Zhao et al., 2017).

    Attributes
    ----------
    data_range : float
        Difference between the largest and smallest possible pixel value, such as 1 for images in [0, 1] or 255.
    window_size : int
        Odd side length of the Gaussian window.
    window_sigma : float
        Standard deviation of the Gaussian window in pixels.

    Raises
    ------
    ValueError
        If data_range or window_sigma is not positive, or window_size is not an odd number of at least 3.
    """

    data_range: float = 1.0
    window_size: int = 11
    window_sigma: float = 1.5

    def __post_init__(self) -> None:
        if self.data_range <= 0.0:
            raise ValueError(f"data_range must be positive, got {self.data_range}")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd number of at least 3, got {self.window_size}")
        if self.window_sigma <= 0.0:
            raise ValueError(f"window_sigma must be positive, got {self.window_sigma}")

    @override
    def _validate_size(self, height: int, width: int) -> None:
        if height < self.window_size or width < self.window_size:
            raise ValueError(f"images must be at least {self.window_size} x {self.window_size}, got {height} x {width}")

    @override
    def _compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        channels: int = predictions.shape[1]
        offsets: Tensor = torch.arange(self.window_size, dtype=predictions.dtype, device=predictions.device)
        profile: Tensor = torch.exp(-0.5 * ((offsets - (self.window_size - 1) / 2.0) / self.window_sigma).pow(2))
        profile = profile / profile.sum()
        window: Tensor = (profile.unsqueeze(1) * profile.unsqueeze(0)).expand(channels, 1, -1, -1)

        def blur(images: Tensor) -> Tensor:
            return functional.conv2d(images, window, groups=channels)

        stability_luminance: float = (0.01 * self.data_range) ** 2
        stability_contrast: float = (0.03 * self.data_range) ** 2
        prediction_means: Tensor = blur(predictions)
        target_means: Tensor = blur(targets)
        prediction_variances: Tensor = blur(predictions.pow(2)) - prediction_means.pow(2)
        target_variances: Tensor = blur(targets.pow(2)) - target_means.pow(2)
        covariances: Tensor = blur(predictions * targets) - prediction_means * target_means
        luminance: Tensor = (2.0 * prediction_means * target_means + stability_luminance) / (
            prediction_means.pow(2) + target_means.pow(2) + stability_luminance
        )
        contrast_structure: Tensor = (2.0 * covariances + stability_contrast) / (
            prediction_variances + target_variances + stability_contrast
        )
        return 1.0 - (luminance * contrast_structure).mean()
