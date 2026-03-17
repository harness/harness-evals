"""Base perturbation abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePerturbation(ABC):
    """Base class for input perturbation generators.

    Perturbation generators produce input variants for robustness testing.
    Each generator takes an input and returns n perturbed versions.
    """

    @abstractmethod
    async def perturb(self, input_text: str, n: int = 5) -> list[str]:
        """Generate n perturbations of the input.

        Args:
            input_text: The original input to perturb.
            n: Number of perturbations to generate.

        Returns:
            A list of perturbed inputs.
        """
        ...
