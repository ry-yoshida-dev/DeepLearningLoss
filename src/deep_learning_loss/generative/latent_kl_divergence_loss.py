from dataclasses import dataclass

import torch
from torch import Tensor

from ..loss import Loss


@dataclass(frozen=True)
class LatentKLDivergenceLoss(Loss):
    """
    KL divergence of a diagonal Gaussian posterior from the standard normal prior; the VAE regularizer.

    The Kullback-Leibler term of the evidence lower bound of a variational autoencoder (Kingma and Welling, 2014),
    ``0.5 * sum(mean ** 2 + variance - 1 - log variance)`` over the latent dimensions, averaged over the batch. Add it
    to a reconstruction loss, weighted by beta for a beta-VAE.

    Attributes
    ----------
    free_bits : float
        Floor in nats on the batch-averaged divergence of each latent dimension (Kingma et al., 2016), which keeps
        dimensions in use against posterior collapse; 0 gives the plain divergence.

    Raises
    ------
    ValueError
        If free_bits is negative.
    """

    free_bits: float = 0.0

    def __post_init__(self) -> None:
        if self.free_bits < 0.0:
            raise ValueError(f"free_bits must be non-negative, got {self.free_bits}")

    def compute(self, means: Tensor, log_variances: Tensor) -> Tensor:
        """
        Compute the divergence summed over the latent dimensions and averaged over the batch.

        Parameters
        ----------
        means : Tensor
            Posterior means with shape ``(batch, *latent)``.
        log_variances : Tensor
            Posterior log-variances with the shape of the means.

        Returns
        -------
        Tensor
            The scalar divergence in nats.

        Raises
        ------
        ValueError
            If the shapes differ or lack a batch axis.
        """
        self._require_same_shape("means", means, "log_variances", log_variances)
        if means.ndim < 2:
            raise ValueError(f"means must have shape (batch, *latent), got {tuple(means.shape)}")
        divergences: Tensor = 0.5 * (means.pow(2) + log_variances.exp() - 1.0 - log_variances)
        dimension_divergences: Tensor = divergences.flatten(1).mean(dim=0)
        return torch.clamp(dimension_divergences, min=self.free_bits).sum()
