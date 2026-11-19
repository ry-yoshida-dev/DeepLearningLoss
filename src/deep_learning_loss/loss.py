import inspect

import torch
from torch import Tensor


class Loss:
    """
    Root of every loss: an objective that maps network outputs and targets to a scalar to minimize.

    Each family below it fixes what the loss compares (values, logits, embeddings, ...), so losses of one family are
    interchangeable.
    """

    @classmethod
    def summarize(cls) -> str:
        """
        Describe in one paragraph what the loss does and when to choose it.

        Returns
        -------
        str
            The first paragraph of the class docstring, on one line.
        """
        documentation: str | None = inspect.getdoc(cls)
        if documentation is None:
            return cls.__name__
        return " ".join(documentation.split("\n\n")[0].split())

    @staticmethod
    def _require_same_shape(first_name: str, first: Tensor, second_name: str, second: Tensor) -> None:
        if first.shape != second.shape:
            raise ValueError(
                f"{first_name} shape {tuple(first.shape)} differs from {second_name} shape {tuple(second.shape)}"
            )

    @staticmethod
    def _require_mask(is_included: Tensor, reference_name: str, reference: Tensor) -> None:
        if is_included.dtype != torch.bool:
            raise TypeError(f"is_included must be a boolean tensor, got {is_included.dtype}")
        Loss._require_same_shape("is_included", is_included, reference_name, reference)

    @staticmethod
    def _require_indices(name: str, indices: Tensor) -> None:
        if indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool:
            raise TypeError(f"{name} must be an integer tensor, got {indices.dtype}")
