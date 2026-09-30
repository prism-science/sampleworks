"""Annealed Langevin SDE sampler for AF3-style models.

Relation to Chroma
------------------
The ``reverse_sde``/``langevin``/``ode`` modes follow Chroma
(github.com/generatebio/chroma):

- ``chroma/layers/structure/diffusion.py``, ``DiffusionChainCov``:
  ``_schedule_coefficients`` (``lambda_t``, ``lambda_langevin``),
  ``reverse_sde``, ``langevin``, ``ode``.
- ``chroma/layers/sde.py``: ``sde_integrate`` (Euler-Maruyama) and
  ``sde_integrate_heun``.
- ``chroma/models/chroma.py``, ``Chroma.sample``: defaults of 500 steps,
  ``inverse_temperature=10`` and ``langevin_factor=2``.

Kept as in Chroma:

- The structure of each ``sde_func``: which terms are present, how
  ``inverse_temperature`` and ``langevin_factor`` enter the drift and the
  noise, and isothermal Langevin (``lambda_langevin = beta``).
- Guidance is tempered together with the model score by default
  (``temper_guidance=True``), because Chroma adds conditioner energies to the
  diffusion energy before taking the gradient.
- Heun's second drift evaluation uses the same ``t`` and the same noise draw.

Rewritten for the VE parameterization the AF3-family wrappers were trained
with (``x_sigma = x_0 + sigma * eps``):

- ``alpha = 1``, so the ``-beta_VP / 2 * X`` drift term is dropped.
- ``g^2 dt = d(sigma^2) = 2 sigma dsigma``, and the noise becomes
  ``sqrt(2 sigma |dsigma|)``.
- The score comes from the denoiser: ``score = -(x_t - x_hat_0) / sigma^2``.
- ``lambda_t`` keeps its form but replaces ``alpha^2`` with
  ``sigma_data^2``, which is exact for Gaussian data ``N(0, sigma_data^2)``.
  Chroma's whitened backbone coordinates have unit scale; all-atom coordinates
  do not.
- Time is discretized on the Karras sigma schedule (as in ``AF3EDMSampler``)
  rather than Chroma's uniform grid in ``t``.

Not reproduced:

- Chroma's backbone-specific correlated noise (``multiply_covariance`` /
  ``_multiply_R``). Noise here is i.i.d. Gaussian and the score is not
  covariance-transformed, since it has no all-atom analogue.
- Chroma's centering. Instead, this sampler centers, randomly rotates and
  aligns to the input exactly like ``AF3EDMSampler``, so that rewards are
  computed in the map frame.

``sde_mode="legacy"`` is the original simplified update. It does not follow
Chroma: its noise is not scaled by ``sigma`` and has no matching pull toward
``x_hat_0``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

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
    sde_mode
        ``"legacy"`` (default): the original two-knob update, where
        ``inverse_temperature`` scales the drift and ``langevin_factor`` scales
        noise of size ``sqrt(|step_scale * dt|)``. ``"reverse_sde"``,
        ``"langevin"``, ``"ode"``: Chroma's ``sde_funcs`` of the same names,
        rewritten for the VE schedule (alpha=1, beta=0, g^2 dt = d(sigma^2)).
        All three share one update, ``x + dt * c * (delta + guidance) +
        sqrt(n) * sqrt(2 sigma |dt|) * z``, built from an optional
        probability-flow ODE term plus ``n`` Langevin units (see
        ``_sde_drift_and_units``). ``"langevin"`` requires
        ``langevin_factor > 0`` and ``"ode"`` requires ``langevin_factor == 0``.
        ``step_scale`` is not applied in these modes.
    align_xt_to_x0
        Rigidly align the noisy state onto the denoised prediction before the
        update (same operation as ``EDMSamplerConfig.alignment_reverse_diffusion``).
    temper_guidance
        If True (default), ``inverse_temperature`` scales the step-scaler
        guidance term along with the model score. If False, the guidance term
        uses the ``inverse_temperature=1`` coefficient, i.e. only the prior is
        tempered: ``p(x)^beta * p(y|x)``. ``langevin_factor`` still scales it
        in the ``reverse_sde`` and ``langevin`` modes.
    integrate_func
        ``"euler_maruyama"`` (default) or ``"heun"``, Chroma's two
        ``integrate_funcs``. ``"heun"`` follows ``chroma.layers.sde.
        sde_integrate_heun``: it re-evaluates the drift at the Euler prediction
        at the same noise level, reuses the same noise draw, and averages the
        two drifts, so it costs two model calls per step. Requires one of the
        Chroma ``sde_mode`` values (not ``"legacy"``).
    step_scale
        Multiplier on the Euler step size, matching ``EDMSamplerConfig``.
        Only used when ``sde_mode="legacy"``.
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
    sde_mode: Literal["legacy", "reverse_sde", "langevin", "ode"] = "legacy"
    align_xt_to_x0: bool = False
    temper_guidance: bool = True
    integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama"
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
        if self.sde_mode not in ("legacy", "reverse_sde", "langevin", "ode"):
            raise ValueError(
                f"sde_mode ({self.sde_mode}) must be 'legacy', 'reverse_sde', 'langevin' or 'ode'"
            )
        if self.sde_mode == "langevin" and self.langevin_factor == 0:
            raise ValueError("sde_mode='langevin' has no drift or noise when langevin_factor=0")
        if self.sde_mode == "ode" and self.langevin_factor != 0:
            raise ValueError("sde_mode='ode' ignores langevin_factor; set it to 0")
        if self.integrate_func not in ("euler_maruyama", "heun"):
            raise ValueError(
                f"integrate_func ({self.integrate_func}) must be 'euler_maruyama' or 'heun'"
            )
        if self.integrate_func == "heun" and self.sde_mode == "legacy":
            raise ValueError("integrate_func='heun' requires a Chroma sde_mode, not 'legacy'")


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

    def _sde_drift_and_units(
        self, inverse_temperature: float, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        r"""Drift coefficient and Langevin-unit count for the Chroma SDE modes.

        Every mode is a probability-flow ODE term (weight 0 or 1) plus ``n``
        Langevin units. The ODE term contributes ``dt * b_ode * delta``; each
        Langevin unit contributes ``dt * b * delta`` of drift and
        ``sqrt(2 sigma |dt|)`` of noise, which leaves the current noise level
        unchanged at equilibrium. The temperature coefficients follow Chroma's
        ``sde_funcs`` with ``langevin_isothermal=True``:

        ============  ===========  =============================================
        mode          ODE term     Langevin units (count x coefficient)
        ============  ===========  =============================================
        reverse_sde   kappa_t      1 x kappa_t  +  langevin_factor x beta
        langevin      --           langevin_factor x beta
        ode           beta         --
        ============  ===========  =============================================

        with ``kappa_t = beta (sigma^2 + sigma_data^2) / (beta sigma^2 +
        sigma_data^2)``: Chroma's ``lambda_t`` with ``alpha^2`` replaced by
        ``sigma_data^2`` for the VE schedule (see the module docstring).

        Parameters
        ----------
        inverse_temperature
            ``beta`` for this term (the model score, or the guidance term
            when ``temper_guidance=False`` passes 1.0).
        sigma
            Current noise level ``sigma_tm``.

        Returns
        -------
        tuple[torch.Tensor, float]
            ``(c, n)``: the update is ``dt * c * direction`` plus
            ``sqrt(n) * sqrt(2 sigma |dt|) * z`` of noise.
        """
        sigma_data_sq = self.config.sigma_data**2
        kappa = (
            inverse_temperature
            * (sigma**2 + sigma_data_sq)
            / (inverse_temperature * sigma**2 + sigma_data_sq)
        )
        langevin_factor = self.config.langevin_factor
        ode_weight, ode_coefficient, langevin_units = {
            "reverse_sde": (1.0, kappa, [(1.0, kappa), (langevin_factor, inverse_temperature)]),
            "langevin": (0.0, kappa, [(langevin_factor, inverse_temperature)]),
            "ode": (1.0, inverse_temperature, []),
        }[self.config.sde_mode]
        drift_coefficient = ode_weight * ode_coefficient + sum(
            count * coefficient for count, coefficient in langevin_units
        )
        num_units = sum(count for count, _ in langevin_units)
        return torch.as_tensor(drift_coefficient), float(num_units)

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
            (guidance_term, loss). ``guidance_term`` is returned separately
            from ``delta`` (unlike ``AF3EDMSampler``) so ``step()`` can scale
            the two with different coefficients.
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

        return torch.as_tensor(scaled_delta_contribution), loss

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

        working_state, drift_update, noise_scale, denoised, loss = self._drift_at(
            state, model_wrapper, context, scaler, features
        )
        noise = noise_scale * torch.randn_like(working_state)

        if self.config.integrate_func == "euler_maruyama":
            next_state = working_state + drift_update + noise
        else:
            # Chroma's sde_integrate_heun: re-evaluate the drift at the Euler prediction at the
            # *same* t (not t + dt), reuse the same noise, and average the two drifts. Both
            # drifts live in the input-aligned frame, since _drift_at re-aligns the prediction.
            predicted_state = working_state + drift_update + noise
            _, predicted_drift, _, _, _ = self._drift_at(
                predicted_state.detach(), model_wrapper, context, scaler, features
            )
            next_state = working_state + 0.5 * (drift_update + predicted_drift) + noise

        return SamplerStepOutput(
            state=next_state,
            denoised=denoised,
            loss=loss,
            log_proposal_correction=None,
        )

    def _drift_at(
        self,
        state: Float[torch.Tensor, "*batch atoms 3"],
        model_wrapper: FlowModelWrapper,
        context: StepParams,
        scaler: StepScalerProtocol | None,
        features: GenerativeModelInput | None,
    ) -> tuple[
        Float[torch.Tensor, "*batch atoms 3"],
        Float[torch.Tensor, "*batch atoms 3"],
        torch.Tensor,
        Float[torch.Tensor, "*batch atoms 3"],
        torch.Tensor | None,
    ]:
        r"""Evaluate the model at ``state`` and return the drift for one step.

        Runs augmentation, the denoiser, input alignment and guidance, then
        builds the drift of the configured ``sde_mode``. The noise is left to
        the caller so it can be shared between Heun's two evaluations.

        Parameters
        ----------
        state
            Coordinates to evaluate at.
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
        tuple
            ``(working_state, drift, noise_scale, denoised, loss)``:
            ``state`` in the input-aligned working frame, the drift to add to
            it, the scalar noise standard deviation for this step, the aligned
            denoised prediction, and the guidance loss (``None`` without a
            scaler).
        """
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

        if self.config.align_xt_to_x0:
            noisy_state_working_frame_t = torch.as_tensor(
                weighted_rigid_align_differentiable(
                    noisy_state_working_frame_t,
                    x_hat_0_working_frame_t,  # target frame
                    weights=torch.ones_like(x_hat_0_working_frame_t[..., 0]),
                    mask=torch.ones_like(x_hat_0_working_frame_t[..., 0]),
                    allow_gradients=False,
                )
            )

        # Tweedie-derived score-following direction: delta = -sigma * score(x, sigma).
        delta = torch.as_tensor((noisy_state_working_frame_t - x_hat_0_working_frame_t) / t_hat)

        loss = None
        guidance_term = torch.zeros_like(delta)
        if scaler is not None:
            guidance_term, loss = self._apply_scaler_guidance(
                scaler=scaler,
                x_hat_0_working_frame=x_hat_0_working_frame_t,
                noisy_state=noisy_state,
                delta=torch.as_tensor(delta),
                context=context,
                model_wrapper=model_wrapper,
                align_transform=align_transform,
                allow_gradients=allow_gradients,
            )

        beta = self.config.inverse_temperature
        guidance_beta = beta if self.config.temper_guidance else 1.0

        if self.config.sde_mode == "legacy":
            # Temperature-scaled deterministic drift, plus injected Langevin diffusion noise.
            step_size = self.config.step_scale * dt  # ty: ignore[unsupported-operator]
            drift_update = step_size * (beta * delta + guidance_beta * guidance_term)
            noise_scale = self.config.langevin_factor * torch.sqrt(
                torch.abs(torch.as_tensor(step_size))
            )
        else:
            # VE form of Chroma's sde_funcs: g^2 dt = 2 sigma dsigma, score = -delta / sigma.
            sigma = torch.as_tensor(t_hat)
            dsigma = torch.as_tensor(dt)
            prior_coefficient, num_langevin_units = self._sde_drift_and_units(beta, sigma)
            guidance_coefficient, _ = self._sde_drift_and_units(guidance_beta, sigma)
            drift_update = dsigma * (
                prior_coefficient * delta + guidance_coefficient * guidance_term
            )
            noise_scale = num_langevin_units**0.5 * torch.sqrt(2.0 * sigma * torch.abs(dsigma))

        return (
            noisy_state_working_frame_t,
            drift_update,
            torch.as_tensor(noise_scale),
            x_hat_0_working_frame_t,
            loss,
        )
