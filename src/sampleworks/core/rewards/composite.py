"""Weighted combination of several reward functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int
from sampleworks.core.rewards.protocol import prepare_reward_if_needed, RewardFunctionProtocol


if TYPE_CHECKING:
    from sampleworks.core.rewards.protocol import RewardInputs


class CompositeReward:
    """Sum of weighted reward functions, itself a reward function.

    Combines terms that score different things about the same coordinates -- a
    density fit and a physical-plausibility prior, say, or two experimental data
    sets. Every term follows the package's sign convention (see
    :class:`~sampleworks.core.rewards.protocol.RewardFunctionProtocol`): values are
    minimized, so the weighted sum is too.

    Weights are the terms' relative influence on the gradient. They default to
    ``1/len(rewards)``, which keeps the combined magnitude comparable to a single
    reward's and so leaves the guidance step size meaning what it meant before.

    Parameters
    ----------
    rewards
        Reward functions to combine. Must not be empty.
    weights
        One weight per reward, or None (default) for uniform ``1/N`` weights.

    Raises
    ------
    ValueError
        If ``rewards`` is empty, ``weights`` has a different length, or any
        weight is negative.
    """

    def __init__(
        self,
        rewards: Sequence[RewardFunctionProtocol],
        weights: Sequence[float] | None = None,
    ):
        if not rewards:
            raise ValueError(
                "CompositeReward needs at least one reward function; combining none of "
                "them has no meaningful value or gradient."
            )

        if weights is None:
            weights = [1.0 / len(rewards)] * len(rewards)
        elif len(weights) != len(rewards):
            raise ValueError(
                f"Got {len(weights)} weights for {len(rewards)} rewards; they must correspond "
                "one to one."
            )

        negative = [w for w in weights if w < 0]
        if negative:
            raise ValueError(
                f"Reward weights must be non-negative, got {negative}. A negative weight "
                "inverts that term's sign and steers away from it."
            )

        self.rewards = list(rewards)
        self.weights = [float(w) for w in weights]

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute the weighted sum of the component rewards.

        Parameters
        ----------
        coordinates
            Atomic coordinates, shape [batch, n_atoms, 3].
        elements
            Atomic element indices, shape [batch, n_atoms].
        b_factors
            Per-atom B-factors, shape [batch, n_atoms].
        occupancies
            Per-atom occupancies, shape [batch, n_atoms].
        unique_combinations
            Pre-computed unique (element, b_factor) pairs, forwarded verbatim.
            Rewards that do not use them ignore them; they exist so a caller can
            hoist that deduplication out of a vmap, where dynamic shapes are not
            allowed.
        inverse_indices
            Indices reconstructing the per-atom values from
            ``unique_combinations``, forwarded verbatim.

        Returns
        -------
        Float[torch.Tensor, ""]
            Scalar value to be minimized.
        """
        total = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
        for reward, weight in zip(self.rewards, self.weights, strict=True):
            total = total + weight * reward(
                coordinates,
                elements,
                b_factors,
                occupancies,
                unique_combinations,
                inverse_indices,
            )
        return total

    def prepare(self, reward_inputs: RewardInputs, *, device: torch.device | str = "cpu") -> None:
        """Prepare each component reward that needs binding to the reward inputs.

        Parameters
        ----------
        reward_inputs
            Model-order reward inputs every component's ``__call__`` will be fed.
        device
            PyTorch device the prepared state is placed on.
        """
        for reward in self.rewards:
            prepare_reward_if_needed(reward, reward_inputs, device=device)

    def __repr__(self) -> str:
        terms = ", ".join(
            f"{weight:g}*{type(reward).__name__}"
            for reward, weight in zip(self.rewards, self.weights, strict=True)
        )
        return f"CompositeReward({terms})"
