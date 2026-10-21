"""
Protocols for scalers. Two levels of abstraction - step-level intervention and trajectory-level
intervention. Examples of the first include training-free guidance, DPS, DMAP, simple gradient steps
down the loss, etc. The second level includes methods that do resampling or modify the sampling
trajectory itself, such as sequential Monte Carlo methods, Feynman-Kaç Steering, DriftLite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING, TypeVar

from jaxtyping import Float

from sampleworks.models.protocol import (
    EnergyBasedModelWrapper,
    FlowModelWrapper,
    FlowOrEnergyBasedModelOutputT,
    StructureModelOutputT,
    StructureModelWrapper,
)
from sampleworks.utils.framework_utils import Array


if TYPE_CHECKING:
    from sampleworks.core.losses.protocol import LossFunctionProtocol
    from sampleworks.core.samplers.protocol import StepParams, TrajectorySampler


ModelOutputT = FlowOrEnergyBasedModelOutputT | StructureModelOutputT
ModelWrapperT = TypeVar(
    "ModelWrapperT",
    FlowModelWrapper,
    EnergyBasedModelWrapper,
    StructureModelWrapper,
)


@dataclass(frozen=True)
class GuidanceOutput:
    """Standardized output from scaler guidance methods."""

    structure: dict
    final_state: Float[Array, "*batch atoms 3"]
    trajectory: Sequence[Float[Array, ...]] | None = None
    losses: list[float | None] | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)


@runtime_checkable
class StepScalerProtocol(Protocol[ModelWrapperT]):
    """Protocol for guidance that modifies the model output.

    StepScalers operate on a per-step basis, computing guidance directions
    from the loss function gradient. Loss function and inputs are
    obtained from StepParams.
    """

    def scale(
        self,
        state: ModelOutputT,
        context: StepParams,
        *,
        model: ModelWrapperT | None = None,
    ) -> tuple[ModelOutputT, Float[Array, " batch"]]:
        r"""Compute guidance direction and loss value.

        Parameters
        ----------
        state
            Current model output (e.g., :math:`\hat{x}_\theta` prediction).
        context
            Step context containing t, dt, and loss/loss_inputs.
        model
            Optional model wrapper for noise-space guidance requiring
            backprop through model.

        Returns
        -------
        tuple[ModelOutputT, Float[Array, " batch"]]
            Guidance direction tensor and the scalar loss value.
        """
        ...

    def guidance_strength(self, context: StepParams) -> Float[Array, " batch"]:
        """Timestep-dependent guidance weight."""
        ...


@runtime_checkable
class TrajectoryScalerProtocol(Protocol):
    """Protocol for trajectory-level guidance that controls the sampling process.

    TrajectoryScalers operate at the population level, using techniques like
    resampling (FK steering) or drift correction (DriftLite) to steer the
    sampling trajectory.
    """

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol[FlowModelWrapper],
        loss_function: LossFunctionProtocol,
        num_particles: int = 1,
    ) -> GuidanceOutput:
        """Generate samples using trajectory level scaling methods. Defines the loop around
        the sampler and score scaler used to produce guidance output.

        Parameters
        ----------
        structure: dict
            Input structure information in the form of an atomworks dictionary.
        model: FlowModelWrapper
            FlowModelWrapper to use for sampling from with sampler.
        sampler: TrajectorySampler
            Sampler to use for generating the trajectory, e.g. EDM solver, Annealed Langevin, etc.
        step_scaler: StepScalerProtocol
            StepScalerProtocol defining update rule for per-step guidance.
        loss: LossFunctionProtocol
            Loss function for steering the model.
        num_particles: int
            Number of particles for trajectory-level methods that require a population.
            Default is 1 (no population)."""
        ...


# TODO: extend protocols for other ModelWrapper types
