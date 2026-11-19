from abc import ABC, abstractmethod

from torch import Tensor

from ..loss import Loss


class AdversarialLoss(Loss, ABC):
    """
    Pair of losses for the discriminator and the generator of a generative adversarial network.

    The discriminator (critic) scores real and generated samples; it learns to tell them apart while the generator
    learns to make its samples score as real. The scores may have any shape, such as one per image or one per patch.
    """

    @abstractmethod
    def compute_discriminator(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        """
        Compute the loss that trains the discriminator.

        Parameters
        ----------
        real_scores : Tensor
            Discriminator outputs for real samples.
        fake_scores : Tensor
            Discriminator outputs for generated samples, detached from the generator.

        Returns
        -------
        Tensor
            The scalar loss.
        """

    @abstractmethod
    def compute_generator(self, fake_scores: Tensor) -> Tensor:
        """
        Compute the loss that trains the generator.

        Parameters
        ----------
        fake_scores : Tensor
            Discriminator outputs for generated samples.

        Returns
        -------
        Tensor
            The scalar loss.
        """
