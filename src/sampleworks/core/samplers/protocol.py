"""Unified sampler protocol for generative models and optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, runtime_checkable, TYPE_CHECKING, TypeVar

import torch
from jaxtyping import Float

from sampleworks.core.losses.protocol import LossFunctionProtocol
from sampleworks.models.protocol import (
    EnergyBasedModelWrapper,
    FlowModelWrapper,
    GenerativeModelInput,
    StructureModelWrapper,
)
from sampleworks.utils.framework_utils import Array


if TYPE_CHECKING:
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.scalers.protocol import StepScalerProtocol
    from sampleworks.utils.atom_reconciler import AtomReconciler


@dataclass(frozen=True, slots=True)
class SamplerSchedule:
    """Base schedule object returned by trajectory-based Sampler implementations.

    A schedule is created once via `compute_schedule(num_steps)` and then passed
    to `get_context_for_step(timestep, schedule)` to build StepParams.

    Implementations should:
    - be immutable dataclasses (frozen=True)
    - store any per-step arrays needed to compute step context
    - provide `as_dict()` for compatibility/serialization.
    """

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class StepParams:
    """Universal step parameters passed to all samplers.

    Contains metadata needed to execute a single sampling step.
    Different sampler types use different fields.
    """

    step_index: int
    total_steps: int | None = None

    # Trajectory parameters (diffusion/flow-matching)
    t: Float[Array, " batch"] | None = None  # Current time/noise level
    dt: Float[Array, " batch"] | None = None  # Step size in time
    # Scale of noise added to inputs (diffusion). Typically is the std. dev. of the noise at this t.
    noise_scale: Float[Array, " batch"] | None = None

    # Optimization parameters
    learning_rate: Float[Array, " batch"] | None = None

    # Guidance parameters (optional, for guided sampling)
    loss_function: LossFunctionProtocol | None = None
    loss_inputs: LossInputs | None = None

    # Optional atom reconciliation
    reconciler: AtomReconciler | None = None
    # Optional alignment reference coordinates in model space (meaning same number of atoms as used
    # in the model representation, not necessarily the structure). Used for rigid alignment.
    alignment_reference: Float[Array, "*batch atoms 3"] | None = None

    # Optional other metadata (could include velocity for momentum, etc.)
    metadata: dict[str, Any] | None = None

    @property
    def t_effective(self) -> Float[Array, " batch"]:
        """Effective time/noise level for this step.

        For trajectory samplers, returns ``t``.

        Raises
        ------
        ValueError
            If this is an optimization sampler with no time information.
        """
        if self.t is None:
            raise ValueError("StepParams has no time information (optimization sampler)")
        return self.t

    @property
    def is_trajectory(self) -> bool:
        """Whether this is a trajectory-based step (diffusion/flow)."""
        return self.t is not None

    def with_loss(
        self,
        loss_function: LossFunctionProtocol,
        loss_inputs: LossInputs,
    ) -> StepParams:
        """Return a new ``StepParams`` with loss information attached.

        Parameters
        ----------
        loss_function : LossFunctionProtocol
            Loss function used for guided sampling.
        loss_inputs : LossInputs
            Pre-computed inputs required by *loss_function* (e.g. reference
            coordinates, masks, density maps).

        Returns
        -------
        StepParams
            Shallow copy with *loss_function* and *loss_inputs* set.
        """
        return replace(self, loss_function=loss_function, loss_inputs=loss_inputs)

    def with_reconciler(
        self,
        reconciler: AtomReconciler,
        alignment_reference: Float[Array, "*batch atoms 3"] | None = None,
    ) -> StepParams:
        """Return a new ``StepParams`` with atom reconciliation context.

        Parameters
        ----------
        reconciler : AtomReconciler
            Adapter that translates between model and structure atom spaces.
        alignment_reference : Tensor or jax.Array, optional
            Reference coordinates for rigid alignment during
            sampling. Shape ``(*batch, n_atoms, 3)``.

        Returns
        -------
        StepParams
            Shallow copy with *reconciler* and *alignment_reference* set.
        """
        return replace(self, reconciler=reconciler, alignment_reference=alignment_reference)

    def with_metadata(
        self,
        metadata: dict[str, Any],
    ) -> StepParams:
        """Return a new ``StepParams`` with updated metadata.

        Entries in *metadata* are merged into any existing metadata dict.
        Conflicting keys are overwritten by the new values.

        Parameters
        ----------
        metadata : dict[str, Any]
            Key/value pairs to merge into the metadata.

        Returns
        -------
        StepParams
            Shallow copy with merged metadata.
        """
        merged_metadata = dict(self.metadata) if self.metadata is not None else {}
        merged_metadata.update(metadata)
        return replace(self, metadata=merged_metadata)


StateT = TypeVar("StateT")
ScheduleT = TypeVar("ScheduleT", bound=SamplerSchedule)
ModelWrapperT = TypeVar(
    "ModelWrapperT",
    FlowModelWrapper,
    EnergyBasedModelWrapper,
    StructureModelWrapper,
    contravariant=True,
)


@dataclass(frozen=True, slots=True)
class SamplerStepOutput(Generic[StateT]):  # noqa: UP046
    """Output from a single sampler step.

    Encapsulates the updated state along with optional intermediate values
    useful for trajectory tracking and debugging.
    """

    state: StateT
    """Updated coordinates after the sampling step."""

    denoised: StateT | None = None
    r"""Denoised prediction :math:`\hat{x}_\theta` from this step, if available."""

    loss: Float[Array, " batch"] | None = None
    """Loss value from scaler, if guidance was applied."""

    log_proposal_correction: Float[Array, " batch"] | None = None
    r"""Log-ratio of base to guided proposal densities for trajectory-based resampling:
    :math:`\log q_{\text{base}}(x_{t+1}|x_t) - \log q_{\text{guided}}(x_{t+1}|x_t)`.

    None if: deterministic step, no guidance applied, or correction not computable."""


@runtime_checkable
class Sampler(Protocol[StateT, ModelWrapperT]):
    """Unified sampler protocol for all sampling strategies.

    The sampler is STATELESS - it receives all information via arguments
    and returns updated state.

    This protocol supports:
    - Trajectory-based methods (diffusion, flow-matching)
    - Optimization-based methods (gradient descent, equilibrium sampling)
    """

    def check_context(self, context: StepParams) -> None:
        """Validate that the provided StepParams is ready for step.

        Raises
        ------
        ValueError
            If the context is incompatible with this sampler.
        """
        ...

    def step(
        self,
        state: StateT,
        model_wrapper: ModelWrapperT,
        context: StepParams,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput[StateT]:
        """Take a single step.

        Parameters
        ----------
        state : StateT
            Current state (coordinates, model inputs, latents, etc.)
        model_wrapper : FlowModelWrapper | EnergyBasedModelWrapper | StructureModelWrapper
            Model wrapper for performing model evaluations.
        context : StepParams
            Step metadata (time, learning rate, loss/loss_inputs, etc.)
        scaler : StepScalerProtocol | None
            Optional step scaler to modify sampler step using loss from context.
        features : GenerativeModelInput | None
            Additional model features/inputs, if any.

        Returns
        -------
        SamplerStepOutput[StateT]
            Output containing updated state, and optionally the denoised prediction, and
            loss.
        """
        ...


@runtime_checkable
class TrajectorySampler(Sampler[StateT, FlowModelWrapper], Protocol):
    """Sampler that integrates a trajectory (diffusion, flow-matching).

    Extends the base Sampler with schedule management for time-based sampling.
    """

    def compute_schedule(self, num_steps: int) -> SamplerSchedule:
        """Compute the full sampling schedule.

        Parameters
        ----------
        num_steps: int
            Number of sampling steps.

        Returns
        -------
        SamplerSchedule
            Schedule object with sampler-specific arrays.
        """
        ...

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the provided schedule is compatible with this sampler.

        Raises
        ------
        ValueError
            If the schedule is incompatible with this sampler.
        """
        ...

    def get_context_for_step(self, step_index: int, schedule: SamplerSchedule) -> StepParams:
        """Build StepParams from schedule for given step.

        Parameters
        ----------
        step_index: int
            Current timestep index (0-indexed).
        schedule: SamplerSchedule
            The schedule returned by compute_schedule().

        Returns
        -------
        StepParams
            Step parameters with t, dt populated for this step.
        """
        ...
