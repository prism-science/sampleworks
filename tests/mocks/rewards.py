"""Mock implementations of reward protocols for testing."""

from typing import Any

import torch
from sampleworks.core.rewards.protocol import RewardInputs
from torch import Tensor


class MockGradientRewardFunction:
    """Mock RewardFunction with predictable gradient behavior for testing.

    Returns a loss where the gradient with respect to coordinates is
    simply the coordinates themselves (times a scale factor).
    """

    def __init__(self, gradient_scale: float = 1.0):
        self.gradient_scale = gradient_scale

    def __call__(
        self,
        coordinates: Tensor,
        elements: Tensor | None = None,
        b_factors: Tensor | None = None,
        occupancies: Tensor | None = None,
        unique_combinations: Tensor | None = None,
        inverse_indices: Tensor | None = None,
    ) -> Tensor:
        """Loss = 0.5 * scale * ||coords||^2, so grad = scale * coords."""
        return 0.5 * self.gradient_scale * (coordinates**2).sum()


class MockPreparableRewardFunction(MockGradientRewardFunction):
    """Two-phase mock reward that records how and when it was prepared.

    Scores exactly like :class:`MockGradientRewardFunction` and adds ``prepare``, so
    it satisfies ``PreparableRewardFunctionProtocol``. Preparation changes nothing
    about the score; the point is the record, so tests can assert which reward
    inputs and device the scalers handed over and that no evaluation preceded
    ``prepare``.
    """

    def __init__(self, gradient_scale: float = 1.0):
        """Start with no preparations and no evaluations recorded.

        Parameters
        ----------
        gradient_scale
            Passed through to :class:`MockGradientRewardFunction`.
        """
        super().__init__(gradient_scale)
        self.prepared_with: list[tuple[int, str]] = []
        self.prepared_inputs: list[RewardInputs] = []
        self.calls = 0
        self.calls_before_prepare = 0

    @property
    def prepared_atom_counts(self) -> list[int]:
        """Atom count of every reward input this reward was prepared against, in order."""
        return [n_atoms for n_atoms, _ in self.prepared_with]

    def prepare(self, reward_inputs: RewardInputs, *, device: torch.device | str = "cpu") -> None:
        """Record the reward inputs (by object and atom count) and device this reward was bound to.

        Parameters
        ----------
        reward_inputs
            Model-order reward inputs the caller is binding the reward to.
        device
            Device the prepared state would be placed on.
        """
        self.prepared_with.append((reward_inputs.b_factors.shape[-1], str(device)))
        self.prepared_inputs.append(reward_inputs)

    def __call__(self, coordinates: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Score as the parent does, counting evaluations and any that precede ``prepare``.

        Parameters
        ----------
        coordinates
            Coordinates to score; forwarded unchanged.
        *args, **kwargs
            Forwarded to :meth:`MockGradientRewardFunction.__call__`.

        Returns
        -------
        Tensor
            ``0.5 * gradient_scale * ||coordinates||^2``.
        """
        self.calls += 1
        if not self.prepared_with:
            self.calls_before_prepare += 1
        return super().__call__(coordinates, *args, **kwargs)
