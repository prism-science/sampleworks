"""Weighted combination of several loss functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

from sampleworks.core.losses.protocol import LossFunctionProtocol, prepare_loss_if_needed


if TYPE_CHECKING:
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.losses.weight_schedule import WeightSchedule


class CompositeLoss:
    """Sum of weighted loss functions, itself a loss function.

    Combines terms that score different things about the same coordinates -- a
    density fit and a physical-plausibility prior, say, or two experimental data
    sets. Every term follows the package's sign convention (see
    :class:`~sampleworks.core.losses.protocol.LossFunctionProtocol`): values are
    minimized, so the weighted sum is too.

    Weights are the terms' relative influence on the gradient. They default to
    ``1/len(losses)``, which keeps the combined magnitude comparable to a single
    loss's and so leaves the guidance step size meaning what it meant before.

    A term with a :class:`~sampleworks.core.losses.weight_schedule.WeightSchedule`
    contributes only the scheduled fraction of its weight. Trajectory scalers ask
    for that through :meth:`at_progress` at every guided step; calling a scheduled
    combination directly scores it at full weight.

    Parameters
    ----------
    losses
        Loss functions to combine. Must not be empty.
    weights
        One weight per loss, or None (default) for uniform ``1/N`` weights.
    schedules
        One schedule per loss, None for a term that always applies at full
        weight, or None (default) to schedule no term.

    Raises
    ------
    ValueError
        If ``losses`` is empty, ``weights`` or ``schedules`` has a different
        length, or any weight is negative.
    """

    def __init__(
        self,
        losses: Sequence[LossFunctionProtocol],
        weights: Sequence[float] | None = None,
        schedules: Sequence[WeightSchedule | None] | None = None,
    ):
        if not losses:
            raise ValueError(
                "CompositeLoss needs at least one loss function; combining none of "
                "them has no meaningful value or gradient."
            )

        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        elif len(weights) != len(losses):
            raise ValueError(
                f"Got {len(weights)} weights for {len(losses)} losses; they must correspond "
                "one to one."
            )

        negative = [w for w in weights if w < 0]
        if negative:
            raise ValueError(
                f"Loss weights must be non-negative, got {negative}. A negative weight "
                "inverts that term's sign and steers away from it."
            )

        if schedules is None:
            schedules = [None] * len(losses)
        elif len(schedules) != len(losses):
            raise ValueError(
                f"Got {len(schedules)} weight schedules for {len(losses)} losses; they must "
                "correspond one to one."
            )

        self.losses = list(losses)
        self.weights = [float(w) for w in weights]
        self.schedules = list(schedules)

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute the weighted sum of the component losses.

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
            Losses that do not use them ignore them; they exist so a caller can
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
        for loss, weight in zip(self.losses, self.weights, strict=True):
            total = total + weight * loss(
                coordinates,
                elements,
                b_factors,
                occupancies,
                unique_combinations,
                inverse_indices,
            )
        return total

    def at_progress(self, progress: float) -> CompositeLoss:
        """Return the combination in force at ``progress`` through the guided steps.

        Parameters
        ----------
        progress
            Position in the guided steps, 0 at the first and 1 at the last (see
            :func:`~sampleworks.core.losses.weight_schedule.guidance_progress`).

        Returns
        -------
        CompositeLoss
            ``self`` when no term is scheduled; otherwise an unscheduled
            combination of the same (already prepared) loss objects whose
            scheduled weights are scaled by their schedule's fraction.
        """
        if all(schedule is None for schedule in self.schedules):
            return self
        weights = [
            weight if schedule is None else weight * schedule.fraction(progress)
            for weight, schedule in zip(self.weights, self.schedules, strict=True)
        ]
        return CompositeLoss(self.losses, weights)

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Prepare each component loss that needs binding to the loss inputs.

        Parameters
        ----------
        loss_inputs
            Model-order loss inputs every component's ``__call__`` will be fed.
        device
            PyTorch device the prepared state is placed on.
        """
        for loss in self.losses:
            prepare_loss_if_needed(loss, loss_inputs, device=device)

    def __repr__(self) -> str:
        terms = ", ".join(
            f"{weight:g}*{type(loss).__name__}"
            + ("" if schedule is None else f"[{schedule.start:g}->{schedule.end:g}]")
            for loss, weight, schedule in zip(
                self.losses, self.weights, self.schedules, strict=True
            )
        )
        return f"CompositeLoss({terms})"
