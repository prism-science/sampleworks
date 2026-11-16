"""Tests of sampler ``step()`` against a Gaussian target with a closed-form denoiser.

For data ``N(0, sigma_data^2 I)`` noised as ``x_sigma = x_0 + sigma * eps``, the marginal at
noise level ``sigma`` is ``N(0, (sigma_data^2 + sigma^2) I)`` and the optimal denoiser is linear,
so every sampler update maps a Gaussian to a Gaussian whose variance is known exactly.
Augmentation and input alignment are switched off: they are rigid motions, which leave an
isotropic Gaussian unchanged, and the tests are about the update rule.
"""

from typing import Literal

import pytest
import torch
from jaxtyping import Float
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig
from sampleworks.core.samplers.protocol import StepParams

from tests.mocks.model_wrappers import GaussianDenoiserWrapper


BATCH_SIZE = 4
NUM_ATOMS = 16384
# The sample variance of BATCH_SIZE * NUM_ATOMS * 3 Gaussian draws has a relative standard
# deviation of about 0.3%.
FIXED_POINT_RTOL = 0.02
# Adds the O(1 / num_steps) discretization error of TRAJECTORY_STEPS steps, at most 1.3%.
TRAJECTORY_RTOL = 0.03
TRAJECTORY_STEPS = 500
FIXED_POINT_ITERATIONS = 100
# Fraction of the distance to equilibrium that one Langevin step removes.
LANGEVIN_RELAXATION = 0.2

SIGMA_DATA_SCALES = [1.0, 16.0]


def repeat_step(
    sampler: AnnealedLangevinSampler | AF3EDMSampler,
    denoiser: GaussianDenoiserWrapper,
    context: StepParams,
) -> Float[torch.Tensor, "batch atoms 3"]:
    """Apply one step context repeatedly, starting from all-zero coordinates.

    Parameters
    ----------
    sampler
        Sampler whose ``step()`` is iterated.
    denoiser
        Analytic denoiser for the Gaussian target.
    context
        Step context reused for every iteration, so the noise level never changes.

    Returns
    -------
    torch.Tensor
        State after ``FIXED_POINT_ITERATIONS`` steps, shape ``(BATCH_SIZE, NUM_ATOMS, 3)``.
    """
    state = torch.zeros(BATCH_SIZE, NUM_ATOMS, 3)
    for _ in range(FIXED_POINT_ITERATIONS):
        state = sampler.step(state, denoiser, context).state
    return state


def run_trajectory(
    sampler: AnnealedLangevinSampler | AF3EDMSampler,
    denoiser: GaussianDenoiserWrapper,
) -> Float[torch.Tensor, "batch atoms 3"]:
    """Sample a full trajectory from the prior at the schedule's largest noise level.

    Parameters
    ----------
    sampler
        Sampler providing the schedule and the update.
    denoiser
        Analytic denoiser for the Gaussian target.

    Returns
    -------
    torch.Tensor
        Final state after ``TRAJECTORY_STEPS`` steps, shape ``(BATCH_SIZE, NUM_ATOMS, 3)``.
    """
    schedule = sampler.compute_schedule(TRAJECTORY_STEPS)
    state = schedule.sigma_tm[0] * denoiser.initialize_from_prior(BATCH_SIZE)
    for step_index in range(TRAJECTORY_STEPS):
        context = sampler.get_context_for_step(step_index, schedule)
        state = sampler.step(state, denoiser, context).state
    return state


@pytest.mark.parametrize("sigma_data", SIGMA_DATA_SCALES)
@pytest.mark.parametrize("noise_level_ratio", [0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    ("inverse_temperature", "langevin_factor"), [(1.0, 1.0), (0.5, 3.0), (6.0, 3.0)]
)
def test_langevin_step_equilibrates_to_tempered_marginal(
    sigma_data: float,
    noise_level_ratio: float,
    inverse_temperature: float,
    langevin_factor: float,
) -> None:
    """Langevin steps at a fixed noise level settle on the marginal raised to the power beta.

    The marginal ``N(0, V)`` with ``V = sigma_data^2 + sigma^2`` raised to ``beta`` has variance
    ``V / beta``. Euler-Maruyama removing a fraction ``r`` of the distance to equilibrium per
    step overshoots it by exactly ``1 / (1 - r / 2)``.
    """
    torch.manual_seed(0)
    sampler = AnnealedLangevinSampler(
        LangevinSamplerConfig(
            sigma_data=sigma_data,
            inverse_temperature=inverse_temperature,
            langevin_factor=langevin_factor,
            sde_mode="langevin",
            augmentation=False,
            align_to_input=False,
        )
    )
    sigma = noise_level_ratio * sigma_data
    marginal_variance = sigma_data**2 + sigma**2
    # langevin_factor units each relax by beta * sigma * |dsigma| / V per step.
    dsigma = (
        -LANGEVIN_RELAXATION * marginal_variance / (langevin_factor * inverse_temperature * sigma)
    )
    context = StepParams(
        step_index=0, total_steps=1, t=torch.tensor(sigma), dt=torch.tensor(dsigma)
    )

    state = repeat_step(sampler, GaussianDenoiserWrapper(sigma_data, NUM_ATOMS), context)

    expected_variance = marginal_variance / inverse_temperature / (1 - LANGEVIN_RELAXATION / 2)
    assert state.var().item() == pytest.approx(expected_variance, rel=FIXED_POINT_RTOL)


@pytest.mark.parametrize("sigma_data", SIGMA_DATA_SCALES)
@pytest.mark.parametrize("noise_level_ratio", [0.5, 1.0, 10.0])
@pytest.mark.parametrize(
    ("step_scale", "noise_scale"),
    [(EDMSamplerConfig().step_scale, EDMSamplerConfig().noise_scale), (1.0, 1.0)],
    ids=["af3_defaults", "unit_scales"],
)
def test_edm_step_fixed_point_variance(
    sigma_data: float, noise_level_ratio: float, step_scale: float, noise_scale: float
) -> None:
    """Repeating one EDM step settles where its churn noise balances its Euler contraction.

    The step adds noise of variance ``injected`` and then multiplies the coordinates by
    ``contraction = 1 + step_scale * dt * t_hat / (sigma_data^2 + t_hat^2)``, so the variance
    converges to ``injected * contraction^2 / (1 - contraction^2)``.
    """
    torch.manual_seed(0)
    config = EDMSamplerConfig(
        sigma_data=sigma_data,
        step_scale=step_scale,
        noise_scale=noise_scale,
        augmentation=False,
        align_to_input=False,
    )
    sampler = AF3EDMSampler(config)
    schedule = sampler.compute_schedule(200)
    step_index = int(torch.argmin(torch.abs(schedule.sigma_tm - noise_level_ratio * sigma_data)))
    context = sampler.get_context_for_step(step_index, schedule)

    state = repeat_step(sampler, GaussianDenoiserWrapper(sigma_data, NUM_ATOMS), context)

    sigma_tm = schedule.sigma_tm[step_index].item()
    t_hat = sigma_tm * (1 + schedule.gamma[step_index].item())
    dt = schedule.sigma_t[step_index].item() - t_hat
    injected = config.noise_scale**2 * (t_hat**2 - sigma_tm**2)
    contraction = 1 + config.step_scale * dt * t_hat / (sigma_data**2 + t_hat**2)
    expected_variance = injected * contraction**2 / (1 - contraction**2)
    assert injected > 0
    assert state.var().item() == pytest.approx(expected_variance, rel=FIXED_POINT_RTOL)


@pytest.mark.parametrize("sigma_data", SIGMA_DATA_SCALES)
@pytest.mark.parametrize(
    ("sde_mode", "inverse_temperature", "langevin_factor"),
    [
        ("reverse_sde", 1.0, 0.0),
        ("reverse_sde", 1.0, 3.0),
        ("ode", 1.0, 0.0),
        ("reverse_sde", 0.5, 0.0),
        ("reverse_sde", 4.0, 0.0),
    ],
)
def test_langevin_trajectory_reaches_tempered_target_variance(
    sigma_data: float,
    sde_mode: Literal["reverse_sde", "ode"],
    inverse_temperature: float,
    langevin_factor: float,
) -> None:
    """A full trajectory ends at the target variance divided by the inverse temperature.

    Covers the settings whose continuous-time limit is exact for a Gaussian target: any
    ``langevin_factor`` when untempered, and the reverse SDE without extra Langevin units when
    tempered.
    """
    torch.manual_seed(0)
    sampler = AnnealedLangevinSampler(
        LangevinSamplerConfig(
            sigma_data=sigma_data,
            inverse_temperature=inverse_temperature,
            langevin_factor=langevin_factor,
            sde_mode=sde_mode,
            augmentation=False,
            align_to_input=False,
        )
    )

    state = run_trajectory(sampler, GaussianDenoiserWrapper(sigma_data, NUM_ATOMS))

    expected_variance = sigma_data**2 / inverse_temperature
    assert state.var().item() == pytest.approx(expected_variance, rel=TRAJECTORY_RTOL)


@pytest.mark.parametrize("sigma_data", SIGMA_DATA_SCALES)
def test_edm_trajectory_reaches_target_variance(sigma_data: float) -> None:
    """A full Euler trajectory with unit step and noise scales ends at the target variance.

    Churn is off because ``gamma_0`` is applied at every step whatever the step count, so its
    discretization error does not vanish as the number of steps grows.
    """
    torch.manual_seed(0)
    sampler = AF3EDMSampler(
        EDMSamplerConfig(
            sigma_data=sigma_data,
            gamma_0=0.0,
            step_scale=1.0,
            noise_scale=1.0,
            augmentation=False,
            align_to_input=False,
        )
    )

    state = run_trajectory(sampler, GaussianDenoiserWrapper(sigma_data, NUM_ATOMS))

    assert state.var().item() == pytest.approx(sigma_data**2, rel=TRAJECTORY_RTOL)
