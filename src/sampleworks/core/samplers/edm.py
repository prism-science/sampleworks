"""EDM-style sampler using AF3 schedule formula."""

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
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch


if TYPE_CHECKING:
    from sampleworks.core.scalers.protocol import StepScalerProtocol


@dataclass(frozen=True, slots=True)
class EDMSchedule(SamplerSchedule):
    """EDM noise schedule arrays.

    All arrays are indexed by step (length ``num_steps``).  Derived quantities
    ``t_hat`` and ``dt`` are precomputed at construction time so that
    ``get_context_for_step`` is a simple lookup.
    """

    sigma_tm: Float[torch.Tensor, " steps"]
    sigma_t: Float[torch.Tensor, " steps"]
    gamma: Float[torch.Tensor, " steps"]
    t_hat: Float[torch.Tensor, " steps"]
    dt: Float[torch.Tensor, " steps"]

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:
        return {
            "sigma_tm": self.sigma_tm,
            "sigma_t": self.sigma_t,
            "gamma": self.gamma,
            "t_hat": self.t_hat,
            "dt": self.dt,
        }


@dataclass(frozen=True, slots=True)
class EDMSamplerConfig:
    r"""Config for an EDM sampler.

    Default values match the AF3 parameterization of the EDM framework
    (Karras et al., 2022) except for ``gamma_min``, which for now is set to 0.2
    instead of 1.0.

    Parameters
    ----------
    sigma_data
        Assumed standard deviation of the training data distribution.
        Used to scale the noise schedule:
        :math:`\sigma(t) = \sigma_\text{data} \cdot
        (s_\max^{1/\rho} + t(s_\min^{1/\rho} - s_\max^{1/\rho}))^\rho`.
    s_max
        Upper bound of the noise schedule in units of :math:`\sigma_\text{data}`.
        Controls the starting (maximum) noise level.
    s_min
        Lower bound of the noise schedule in units of :math:`\sigma_\text{data}`.
        Controls the ending (minimum) noise level.
    p
        Exponent :math:`\rho` controlling the shape of the noise schedule
        interpolation between ``s_max`` and ``s_min``.  Higher values
        concentrate more steps at lower noise levels.  Called ``rho`` in
        Karras et al., Table 5.
    gamma_min
        Minimum sigma threshold below which stochastic noise inflation
        (:math:`\gamma`) is disabled (set to 0).  Prevents adding noise
        at near-clean noise levels where it would dominate the signal.
    gamma_0
        Stochastic noise inflation factor :math:`\gamma` applied at each
        step when :math:`\sigma > \gamma_\min`.  Called ``S_churn``
        (divided by num_steps) in Karras et al., Table 5.
    noise_scale
        Multiplier on the stochastic noise magnitude, corresponding to
        :math:`S_\text{noise}` in Karras et al., Table 5.  Values slightly
        above 1.0 compensate for discretization error.
    step_scale
        Multiplier on the Euler step size :math:`\Delta t`.  Values > 1
        take larger steps (AF3 uses 1.5).
    augmentation
        Whether to apply random SO(3) rotation augmentation + small translation before each
        denoising step.
    align_to_input
        Whether to rigidly align the denoised prediction :math:`\hat{x}_0`
        back to the input reference frame after each step.
    alignment_reverse_diffusion
        Whether to also align the noisy state to the denoised prediction
        before computing the denoising direction. This is from Boltz-1, and doesn't work
        well during inference for non-Boltz models.
    scale_guidance_to_diffusion
        Whether to rescale the guidance direction to match the magnitude
        of the diffusion denoising update.
    device
        Torch device for schedule tensor allocation.
    """

    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0
    gamma_min: float = 0.2  # this is not the default value from AF3! AF3 uses 1.0
    gamma_0: float = 0.8
    noise_scale: float = 1.003
    step_scale: float = 1.5
    augmentation: bool = True
    align_to_input: bool = True
    alignment_reverse_diffusion: bool = False
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


class AF3EDMSampler:
    """EDM-style sampler from AF3-like models.

    Initialized with a single :class:`EDMSamplerConfig` object that holds all
    schedule hyperparameters and runtime options.  Default values in the config
    match the AF3 parameterization, except for ``gamma_min`` which is set to 0.2 instead of 1.0.

    This sampler implements the EDM (Karras et al.) style sampling
    approach as used in AlphaFold3 and related models, which is the Euler
    variant.

    References
    ----------
    Karras et al. "Elucidating the Design Space of Diffusion-Based Generative
    Models" (NeurIPS 2022). https://arxiv.org/abs/2206.00364

    AlphaFold3 Supplementary Information, Algorithm 18 "Sample".
    https://www.nature.com/articles/s41586-024-07487-w
    """

    def __init__(self, config: EDMSamplerConfig) -> None:
        """Initialize the sampler with a configuration object.

        Parameters
        ----------
        config : EDMSamplerConfig
            Configuration object containing all schedule hyperparameters
            (``sigma_data``, ``s_max``, ``s_min``, ``p``, ``gamma_min``,
            ``gamma_0``, ``noise_scale``, ``step_scale``) and runtime flags
            (``augmentation``, ``align_to_input``,
            ``alignment_reverse_diffusion``, ``scale_guidance_to_diffusion``,
            ``device``).
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
            raise ValueError("AF3EDMSampler requires trajectory-based StepParams with time info")
        if (
            context.t is None
            or context.dt is None
            or context.noise_scale is None
            or context.total_steps is None
        ):
            raise ValueError("AF3EDMSampler requires t, dt, and noise_scale in StepParams")
        if context.step_index >= context.total_steps:
            raise ValueError("StepParams step_index exceeds total_steps")

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the provided schedule is compatible with this sampler.

        Raises
        ------
        ValueError
            If the schedule is incompatible with this sampler.
        """
        if not (
            hasattr(schedule, "sigma_tm")
            and hasattr(schedule, "sigma_t")
            and hasattr(schedule, "gamma")
        ):
            raise ValueError(
                "EDMSampler requires SamplerSchedule with sigma_tm, sigma_t, and gamma"
            )

    def compute_schedule(self, num_steps: int) -> EDMSchedule:
        r"""Compute sigma-based schedule using RF3's EDM formula.

        Uses the formula:

        .. math::

            \sigma = \sigma_{\text{data}} \cdot
            \left(s_{\max}^{1/p} + t \cdot (s_{\min}^{1/p} - s_{\max}^{1/p})\right)^p

        where :math:`t` goes from 0 to 1 over ``num_steps``.

        Parameters
        ----------
        num_steps: int
            Number of diffusion sampling steps.

        Returns
        -------
        EDMSchedule
            Schedule object with `sigma_tm`, `sigma_t`, `gamma`, `t_hat`, and `dt` arrays.
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

        gammas = torch.where(
            sigmas > self.config.gamma_min,
            torch.tensor(self.config.gamma_0, device=self.config.device),
            torch.tensor(0.0, device=self.config.device),
        )

        sigma_tm = sigmas[:-1]
        sigma_t = sigmas[1:]
        gamma = gammas[1:]
        t_hat = sigma_tm * (1 + gamma)
        dt = sigma_t - t_hat

        return EDMSchedule(
            sigma_tm=sigma_tm,
            sigma_t=sigma_t,
            gamma=gamma,
            t_hat=t_hat,
            dt=dt,
        )

    def get_context_for_step(self, step_index: int, schedule: SamplerSchedule) -> StepParams:
        """Build StepParams from schedule for given step.

        Parameters
        ----------
        step_index: int
            Current timestep index (0-indexed).
        schedule: SamplerSchedule
            The schedule returned by compute_schedule() (must be SamplerSchedule with `sigma_tm`,
            `sigma_t`, `gamma`, `t_hat`, `dt`).

        Returns
        -------
        StepParams
            Context with t, dt, noise_scale populated for this step.
        """

        self.check_schedule(schedule)

        t_hat = schedule.t_hat[step_index]  # ty: ignore[unresolved-attribute] (accessible after check_schedule)
        dt = schedule.dt[step_index]  # ty: ignore[unresolved-attribute]
        sigma_tm = schedule.sigma_tm[step_index]  # ty: ignore[unresolved-attribute]
        eps_scale = self.config.noise_scale * torch.sqrt(t_hat**2 - sigma_tm**2)

        total_steps = len(schedule.sigma_t)  # ty: ignore[unresolved-attribute] (this will be accessible due to the check above)

        return StepParams(
            step_index=step_index,
            total_steps=total_steps,
            t=t_hat,
            dt=dt,
            noise_scale=eps_scale,
        )

    def _apply_scaler_guidance(
        self,
        scaler: StepScalerProtocol,
        # Denoised prediction in the working frame (after optional augmentation and alignment)
        x_hat_0_working_frame: Float[torch.Tensor, "*batch n 3"],
        noisy_state: Float[torch.Tensor, "*batch n 3"],
        delta: torch.Tensor,
        context: StepParams,
        model_wrapper: FlowModelWrapper,
        align_transform: Mapping[str, torch.Tensor] | None,
        allow_gradients: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Apply guidance from scaler to the denoising direction.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
            (modified_delta, loss, proposal_shift)
        """
        scaler_metadata: dict[str, object] = {"x_t": noisy_state}
        scaler_context = context.with_metadata(scaler_metadata)

        guidance_direction, loss = scaler.scale(
            x_hat_0_working_frame, scaler_context, model=model_wrapper
        )

        # make sure device matches
        guidance_direction = torch.as_tensor(guidance_direction, device=noisy_state.device)
        loss = torch.as_tensor(loss)
        guidance_weight = scaler.guidance_strength(context)

        # Ensure guidance_weight has batch dimension matching guidance_direction
        batch_size = guidance_direction.shape[0]
        guidance_weight = torch.as_tensor(guidance_weight, device=noisy_state.device)
        if guidance_weight.ndim == 0:
            guidance_weight = guidance_weight.unsqueeze(0)
        # expand/match batch dimension of guidance weight to guidance_direction
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
            # scaler handles any adjustment/clipping of guidance direction, but we have diffusion
            # update magnitude here, so can optionally scale to match
            guidance_direction = guidance_direction * delta_norm

        scaled_delta_contribution = (
            einx.multiply("b, b n c -> b n c", guidance_weight, guidance_direction)
            / context.t_effective
        )
        proposal_shift = self.config.step_scale * context.dt * scaled_delta_contribution  # ty: ignore[unsupported-operator] (dt will be Array if check_context didn't raise)

        result = delta + scaled_delta_contribution
        return torch.as_tensor(result), loss, torch.as_tensor(proposal_shift)

    def step(
        self,
        state: Float[torch.Tensor, "*batch num_points 3"],
        model_wrapper: FlowModelWrapper,
        context: StepParams,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput:
        r"""Take EDM diffusion step with optional guidance.

        Based on Supplemental Algorithm 18 from the AlphaFold3 paper.

        Parameters
        ----------
        state
            Current noisy coordinates.
        model_wrapper
            Model wrapper for :math:`\hat{x}_\theta` prediction.
        context
            Step context with t, dt, noise_scale, and optionally loss info.
        scaler
            Optional step scaler for computing guidance from losses.
        features
            Additional model features/inputs.

        Returns
        -------
        SamplerStepOutput
            Output containing updated state, denoised prediction :math:`\hat{x}_\theta`, and loss.
        """
        self.check_context(context)

        t_hat = context.t_effective
        dt = context.dt
        # check_context() guarantees noise_scale is present for sampling steps.
        eps_scale = cast(float, context.noise_scale)
        allow_gradients = True if scaler and getattr(scaler, "requires_gradients", False) else False

        centroid = einx.mean("... [n] c", state)
        state_centered = einx.subtract("... n c, ... c -> ... n c", state, centroid)

        transform = (
            create_random_transform(state_centered, center_before_rotation=False)
            if self.config.augmentation
            else None
        )

        maybe_augmented_state = (
            apply_forward_transform(state_centered, transform, rotation_only=False)
            if transform is not None
            else state_centered
        )

        # Store eps separately for proper frame transformation
        # eps_scale will be float if check_context didn't raise
        eps = torch.randn_like(maybe_augmented_state) * eps_scale
        noisy_state = maybe_augmented_state + eps
        noisy_state = torch.as_tensor(noisy_state).detach().requires_grad_(allow_gradients)

        # t_hat will be float if check_context didn't raise
        # Use no_grad when gradients aren't needed to avoid memory overhead from
        # gradient checkpointing holding intermediate activations
        with torch.set_grad_enabled(allow_gradients):
            x_hat_0 = model_wrapper.step(noisy_state, t_hat, features=features)

        reconciler = (
            context.reconciler.to(torch.as_tensor(x_hat_0).device)
            if context.reconciler is not None
            else None
        )

        # work in augmented frame
        x_hat_0_working_frame = x_hat_0
        noisy_state_working_frame = noisy_state
        eps_working_frame = eps
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

            _, eps_working_frame, noisy_state_working_frame = transform_coords_and_noise_to_frame(
                torch.as_tensor(maybe_augmented_state), torch.as_tensor(eps), align_transform
            )

        if self.config.alignment_reverse_diffusion:
            noisy_state_working_frame = weighted_rigid_align_differentiable(
                torch.as_tensor(noisy_state_working_frame),
                torch.as_tensor(x_hat_0_working_frame),  # <-- this is what is being aligned to
                weights=torch.ones_like(torch.as_tensor(x_hat_0_working_frame)[..., 0]),
                mask=torch.ones_like(torch.as_tensor(x_hat_0_working_frame)[..., 0]),
                allow_gradients=False,
            )

        x_hat_0_working_frame_t = torch.as_tensor(x_hat_0_working_frame)
        noisy_state_working_frame_t = torch.as_tensor(noisy_state_working_frame)
        delta = torch.as_tensor((noisy_state_working_frame_t - x_hat_0_working_frame_t) / t_hat)

        loss = None
        log_proposal_correction = None
        if scaler is not None:
            delta, loss, proposal_shift = self._apply_scaler_guidance(
                scaler=scaler,
                x_hat_0_working_frame=x_hat_0_working_frame_t,
                noisy_state=noisy_state,
                delta=torch.as_tensor(delta),
                context=context,
                model_wrapper=model_wrapper,
                align_transform=align_transform,
                allow_gradients=allow_gradients,
            )

            # eps is the noise added. scaled_guidance_update is the shift.
            # ll_diff = (eps^2 - (eps + shift)^2) / 2var
            # Sum over dimensions to get total log likelihood difference for this particle.
            # TODO: this will need to be altered if we figure out how to vmap over this step for
            # multiple particles.
            # Only compute when noise_var > 0 to avoid division by near-zero
            # (matching Boltz behavior)
            noise_var = eps_scale**2
            if noise_var > 0:
                log_proposal_correction = einx.sum(
                    "... [b n c]", eps_working_frame**2 - (eps_working_frame + proposal_shift) ** 2
                ) / (2 * noise_var)

        # Euler step: x_{t-1} = x_t + step_scale * dt * delta
        # ty sees dt as float | None, but it will be float if check_context didn't raise
        next_state = noisy_state_working_frame_t + self.config.step_scale * dt * delta  # ty: ignore[unsupported-operator]

        return SamplerStepOutput(
            state=next_state,
            denoised=x_hat_0_working_frame_t,
            loss=loss,
            log_proposal_correction=log_proposal_correction,
        )
