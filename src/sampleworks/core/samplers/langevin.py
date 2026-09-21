"""Annealed Langevin SDE sampler for AF3-style models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import einx
import torch
from jaxtyping import Float
from loguru import logger

from sampleworks.core.samplers.protocol import SamplerSchedule, SamplerStepOutput, StepParams
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput
from sampleworks.utils.frame_transforms import (
    align_to_reference_frame,
    apply_forward_transform,
    create_random_transform,
    transform_coords_and_noise_to_frame,
)
from sampleworks.utils.framework_utils import match_batch


if TYPE_CHECKING:
    from sampleworks.core.scalers.protocol import StepScalerProtocol


@dataclass(frozen=True, slots=True)
class LangevinSchedule(SamplerSchedule):
    """Noise schedule for :class:`AnnealedLangevinSampler`.

    Uses the same Karras/EDM sigma(t) formula as ``EDMSchedule`` -- AF3-family
    model wrappers were trained under this noise convention, so the annealing
    schedule must match it regardless of the reverse-time update rule used
    once the model is queried.  Unlike ``EDMSchedule``, there is no
    ``gamma``/churn re-noising step: stochasticity comes entirely from the
    Langevin diffusion term added in ``AnnealedLangevinSampler.step()``, so
    the model is queried directly at ``sigma_tm`` each step (``t_hat ==
    sigma_tm``).
    """

    sigma_tm: Float[torch.Tensor, " steps"]
    sigma_t: Float[torch.Tensor, " steps"]
    t_hat: Float[torch.Tensor, " steps"]
    dt: Float[torch.Tensor, " steps"]

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:
        return {
            "sigma_tm": self.sigma_tm,
            "sigma_t": self.sigma_t,
            "t_hat": self.t_hat,
            "dt": self.dt,
        }


@dataclass(frozen=True, slots=True)
class LangevinSamplerConfig:
    r"""Config for the Annealed Langevin SDE sampler.

    The ``sigma_data``/``s_max``/``s_min``/``p`` schedule parameters have the
    same meaning as in ``EDMSamplerConfig`` and should normally be left at
    their defaults so the schedule matches what the model was trained on.

    Parameters
    ----------
    sigma_data
        Assumed standard deviation of the training data distribution. See
        ``EDMSamplerConfig.sigma_data``.
    s_max
        Upper bound of the noise schedule (starting noise level).
    s_min
        Lower bound of the noise schedule (ending noise level).
    p
        Exponent (``rho`` in Karras et al.) controlling schedule spacing.
    inverse_temperature
        Scales the score's contribution to the drift term (Chroma's
        parameter of the same name). Above 1.0 biases sampling toward
        higher-likelihood, lower-diversity structures; 1.0 is unscaled.
    langevin_factor
        Strength of the Langevin (Brownian) noise injected on top of the
        deterministic drift (Chroma's parameter of the same name). ``0.0``
        is a deterministic Euler step; positive values make it a true SDE.
    step_scale
        Multiplier on the Euler step size, matching ``EDMSamplerConfig``.
    augmentation
        Whether to apply random SO(3) rotation augmentation before each
        denoising step, matching ``EDMSamplerConfig``.
    align_to_input
        Whether to rigidly align the denoised prediction back to the input
        reference frame after each step, matching ``EDMSamplerConfig``.
    scale_guidance_to_diffusion
        Whether to rescale the guidance direction to match the magnitude of
        the base (temperature-scaled score) update, matching
        ``EDMSamplerConfig``.
    device
        Torch device for schedule tensor allocation.
    """

    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0
    inverse_temperature: float = 1.0
    langevin_factor: float = 0.0
    step_scale: float = 1.5
    augmentation: bool = True
    align_to_input: bool = True
    scale_guidance_to_diffusion: bool = True
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.p == 0:
            raise ValueError("p must be nonzero (used as exponent denominator in schedule formula)")
        if self.s_max <= 0 or self.s_min <= 0:
            raise ValueError(f"s_max ({self.s_max}) and s_min ({self.s_min}) must be positive")
        if self.s_min >= self.s_max:
            raise ValueError(f"s_min ({self.s_min}) must be less than s_max ({self.s_max})")
        if self.sigma_data <= 0:
            raise ValueError(f"sigma_data ({self.sigma_data}) must be positive")
        if self.inverse_temperature <= 0:
            raise ValueError(f"inverse_temperature ({self.inverse_temperature}) must be positive")
        if self.langevin_factor < 0:
            raise ValueError(f"langevin_factor ({self.langevin_factor}) must be non-negative")


class AnnealedLangevinSampler:
    """Annealed Langevin SDE sampler for AF3-like models.

    Drop-in ``TrajectorySampler`` alternative to :class:`AF3EDMSampler`
    (``sampleworks.core.samplers.edm``): same schedule shape, same alignment
    and step-scaler-guidance handling, but the final update combines a
    temperature-scaled score-following drift with an injected Langevin
    diffusion term instead of Heun's second-order ODE step.

    ``langevin_factor=0.0`` recovers a purely deterministic, temperature-
    scaled Euler step; increasing it adds stochastic Langevin noise on top,
    analogous to Chroma's parameter of the same name.

    Notes
    -----
    The alignment and scaler-guidance logic below is copied from
    ``AF3EDMSampler`` rather than shared with it. This is a deliberate,
    temporary duplication -- factoring it into a shared helper is deferred
    until this sampler's results have been validated.
    """

    def __init__(self, config: LangevinSamplerConfig) -> None:
        """Initialize the sampler with a configuration object.

        Parameters
        ----------
        config : LangevinSamplerConfig
            See :class:`LangevinSamplerConfig` for field documentation.
        """
        self.config = config

    def check_context(self, context: StepParams) -> None:
        """Validate that the provided StepParams is ready for step.

        Raises
        ------
        ValueError
            If the context is incompatible with this sampler.
        """
        if not context.is_trajectory:
            raise ValueError(
                "AnnealedLangevinSampler requires trajectory-based StepParams with time info"
            )
        if context.t is None or context.dt is None or context.total_steps is None:
            raise ValueError("AnnealedLangevinSampler requires t and dt in StepParams")
        if context.step_index >= context.total_steps:
            raise ValueError("StepParams step_index exceeds total_steps")

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the provided schedule is compatible with this sampler.

        Raises
        ------
        ValueError
            If the schedule is incompatible with this sampler.
        """
        if not (hasattr(schedule, "sigma_tm") and hasattr(schedule, "sigma_t")):
            raise ValueError(
                "AnnealedLangevinSampler requires SamplerSchedule with sigma_tm and sigma_t"
            )

    def compute_schedule(self, num_steps: int) -> LangevinSchedule:
        r"""Compute the sigma-based annealing schedule.

        Uses the same formula as ``AF3EDMSampler.compute_schedule()``:

        .. math::

            \sigma = \sigma_{\text{data}} \cdot
            \left(s_{\max}^{1/p} + t \cdot (s_{\min}^{1/p} - s_{\max}^{1/p})\right)^p

        with no churn/re-noising step (``t_hat == sigma_tm``), since
        stochasticity here comes from the Langevin diffusion term in
        ``step()`` rather than from re-noising between steps.

        Parameters
        ----------
        num_steps : int
            Number of diffusion sampling steps.

        Returns
        -------
        LangevinSchedule
            Schedule object with ``sigma_tm``, ``sigma_t``, ``t_hat``, and
            ``dt`` arrays.
        """
        t_values = torch.linspace(0, 1, num_steps + 1, device=self.config.device)

        sigmas = (
            self.config.sigma_data
            * (
                self.config.s_max ** (1 / self.config.p)
                + t_values
                * (
                    self.config.s_min ** (1 / self.config.p)
                    - self.config.s_max ** (1 / self.config.p)
                )
            )
            ** self.config.p
        )

        sigma_tm = sigmas[:-1]
        sigma_t = sigmas[1:]
        dt = sigma_t - sigma_tm

        return LangevinSchedule(sigma_tm=sigma_tm, sigma_t=sigma_t, t_hat=sigma_tm, dt=dt)

    def get_context_for_step(self, step_index: int, schedule: SamplerSchedule) -> StepParams:
        """Build StepParams from schedule for given step.

        Parameters
        ----------
        step_index : int
            Current timestep index (0-indexed).
        schedule : SamplerSchedule
            The schedule returned by compute_schedule() (must have
            ``sigma_tm``, ``sigma_t``, ``t_hat``, ``dt``).

        Returns
        -------
        StepParams
            Context with ``t`` and ``dt`` populated for this step.
        """
        self.check_schedule(schedule)

        t_hat = schedule.t_hat[step_index]  # ty: ignore[unresolved-attribute]
        dt = schedule.dt[step_index]  # ty: ignore[unresolved-attribute]
        total_steps = len(schedule.sigma_t)  # ty: ignore[unresolved-attribute]

        return StepParams(
            step_index=step_index,
            total_steps=total_steps,
            t=t_hat,
            dt=dt,
            noise_scale=t_hat,
        )

    def _apply_scaler_guidance(
        self,
        scaler: StepScalerProtocol,
        x_hat_0_working_frame: Float[torch.Tensor, "*batch n 3"],
        noisy_state: Float[torch.Tensor, "*batch n 3"],
        delta: torch.Tensor,
        context: StepParams,
        model_wrapper: FlowModelWrapper,
        align_transform: Mapping[str, torch.Tensor] | None,
        allow_gradients: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply guidance from scaler to the denoising direction.

        Copied from ``AF3EDMSampler._apply_scaler_guidance`` (see class
        docstring), minus ``proposal_shift`` -- this sampler doesn't derive
        ``log_proposal_correction`` for its Langevin term, so it isn't
        FKSteering-compatible yet; only tested against PureGuidance/CSG.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            (modified_delta, loss)
        """
        scaler_metadata: dict[str, object] = {"x_t": noisy_state}
        scaler_context = context.with_metadata(scaler_metadata)

        guidance_direction, loss = scaler.scale(
            x_hat_0_working_frame, scaler_context, model=model_wrapper
        )

        guidance_direction = torch.as_tensor(guidance_direction, device=noisy_state.device)
        loss = torch.as_tensor(loss)
        guidance_weight = scaler.guidance_strength(context)

        batch_size = guidance_direction.shape[0]
        guidance_weight = torch.as_tensor(guidance_weight, device=noisy_state.device)
        if guidance_weight.ndim == 0:
            guidance_weight = guidance_weight.unsqueeze(0)
        guidance_weight = torch.as_tensor(
            match_batch(guidance_weight, target_batch_size=batch_size),
            device=noisy_state.device,
        )

        if align_transform is not None and allow_gradients:
            guidance_direction = apply_forward_transform(
                guidance_direction, align_transform, rotation_only=True
            )

        if self.config.scale_guidance_to_diffusion:
            delta_norm = torch.linalg.norm(delta, dim=(-1, -2), keepdim=True)
            guidance_direction = guidance_direction * delta_norm

        scaled_delta_contribution = (
            einx.multiply("b, b n c -> b n c", guidance_weight, guidance_direction)
            / context.t_effective
        )

        result = delta + scaled_delta_contribution
        return torch.as_tensor(result), loss

    def step(
        self,
        state: Float[torch.Tensor, "*batch num_points 3"],
        model_wrapper: FlowModelWrapper,
        context: StepParams,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput:
        r"""Take one Annealed Langevin SDE step with optional guidance.

        The denoised prediction and its Tweedie-derived score-following
        direction (``delta``) are computed exactly as in
        ``AF3EDMSampler.step()``. The final update then differs: instead of
        an EDM-style step, it combines a temperature-scaled deterministic
        drift with an injected Langevin diffusion term (see module/class
        docstrings for the full update equation).

        Parameters
        ----------
        state
            Current noisy coordinates.
        model_wrapper
            Model wrapper for :math:`\hat{x}_\theta` prediction.
        context
            Step context with ``t``, ``dt``, and optionally reward info.
        scaler
            Optional step scaler for computing guidance from rewards.
        features
            Additional model features/inputs.

        Returns
        -------
        SamplerStepOutput
            Output containing updated state, denoised prediction
            :math:`\hat{x}_\theta`, and loss. ``log_proposal_correction`` is
            always ``None`` (see ``_apply_scaler_guidance`` docstring).
        """
        self.check_context(context)

        t_hat = context.t_effective
        dt = context.dt
        allow_gradients = True if scaler and getattr(scaler, "requires_gradients", False) else False

        centroid = einx.mean("... [n] c", state)
        state_centered = einx.subtract("... n c, ... c -> ... n c", state, centroid)

        transform = (
            create_random_transform(state_centered, center_before_rotation=False)
            if self.config.augmentation
            else None
        )

        noisy_state = (
            apply_forward_transform(state_centered, transform, rotation_only=False)
            if transform is not None
            else state_centered
        )
        noisy_state = torch.as_tensor(noisy_state).detach().requires_grad_(allow_gradients)

        with torch.set_grad_enabled(allow_gradients):
            x_hat_0 = model_wrapper.step(noisy_state, t_hat, features=features)

        reconciler = (
            context.reconciler.to(torch.as_tensor(x_hat_0).device)
            if context.reconciler is not None
            else None
        )

        x_hat_0_working_frame = x_hat_0
        noisy_state_working_frame = noisy_state
        align_transform = None
        alignment_reference = (
            torch.as_tensor(context.alignment_reference, device=x_hat_0.device, dtype=x_hat_0.dtype)
            if context.alignment_reference is not None
            else None
        )

        if alignment_reference is not None and x_hat_0.ndim == 3:
            alignment_reference = match_batch(
                alignment_reference,
                target_batch_size=x_hat_0.shape[0],
            )

        if self.config.align_to_input and alignment_reference is None:
            logger.warning(
                "align_to_input is True but no alignment_reference provided; "
                "skipping alignment. Set alignment_reference on StepParams via "
                "with_reconciler() to enable alignment."
            )

        if self.config.align_to_input and alignment_reference is not None:
            if reconciler is not None:
                x_hat_0_working_frame, align_transform = reconciler.align(
                    torch.as_tensor(x_hat_0),
                    alignment_reference,
                    allow_gradients=allow_gradients,
                )
            else:
                x_hat_0_working_frame, align_transform = align_to_reference_frame(
                    torch.as_tensor(x_hat_0),
                    alignment_reference,
                    allow_gradients=allow_gradients,
                )

            _, _, noisy_state_working_frame = transform_coords_and_noise_to_frame(
                torch.as_tensor(noisy_state),
                torch.zeros_like(torch.as_tensor(noisy_state)),
                align_transform,
            )

        x_hat_0_working_frame_t = torch.as_tensor(x_hat_0_working_frame)
        noisy_state_working_frame_t = torch.as_tensor(noisy_state_working_frame)
        # Tweedie-derived score-following direction: delta = -sigma * score(x, sigma).
        delta = torch.as_tensor((noisy_state_working_frame_t - x_hat_0_working_frame_t) / t_hat)

        loss = None
        if scaler is not None:
            delta, loss = self._apply_scaler_guidance(
                scaler=scaler,
                x_hat_0_working_frame=x_hat_0_working_frame_t,
                noisy_state=noisy_state,
                delta=torch.as_tensor(delta),
                context=context,
                model_wrapper=model_wrapper,
                align_transform=align_transform,
                allow_gradients=allow_gradients,
            )

        # Temperature-scaled deterministic drift, plus injected Langevin diffusion noise.
        step_size = self.config.step_scale * dt  # ty: ignore[unsupported-operator]
        drift_update = step_size * self.config.inverse_temperature * delta
        langevin_noise = self.config.langevin_factor * torch.sqrt(
            torch.abs(torch.as_tensor(step_size))
        ) * torch.randn_like(noisy_state_working_frame_t)

        next_state = noisy_state_working_frame_t + drift_update + langevin_noise

        return SamplerStepOutput(
            state=next_state,
            denoised=x_hat_0_working_frame_t,
            loss=loss,
            log_proposal_correction=None,
        )
