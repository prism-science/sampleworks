"""Tests for the Annealed Langevin SDE sampler's schedule and configuration."""

import math

import pytest
import torch
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig
from sampleworks.core.samplers.schedule import NoiseScheduleConfig


NUM_STEPS = 50


@pytest.mark.parametrize(
    "schedule",
    [
        NoiseScheduleConfig(),
        NoiseScheduleConfig(sigma_data=10.3, s_max=80, s_min=0.001),
        NoiseScheduleConfig(p=None),
    ],
    ids=["default", "custom_karras", "geometric"],
)
def test_samplers_share_schedule(schedule: NoiseScheduleConfig) -> None:
    """Both samplers place their noise levels where the shared schedule config says."""
    langevin = AnnealedLangevinSampler(LangevinSamplerConfig(schedule=schedule))
    edm = AF3EDMSampler(EDMSamplerConfig(schedule=schedule))
    langevin_schedule = langevin.compute_schedule(NUM_STEPS)
    edm_schedule = edm.compute_schedule(NUM_STEPS)

    torch.testing.assert_close(langevin_schedule.sigma_tm, edm_schedule.sigma_tm)
    torch.testing.assert_close(langevin_schedule.sigma_t, edm_schedule.sigma_t)
    torch.testing.assert_close(
        langevin_schedule.sigma_tm[0], torch.tensor(schedule.sigma_data * schedule.s_max)
    )


def test_default_schedule_matches_edm_sampler() -> None:
    """Default Langevin and EDM configs share the Karras schedule."""
    langevin = AnnealedLangevinSampler(LangevinSamplerConfig()).compute_schedule(NUM_STEPS)
    edm = AF3EDMSampler(EDMSamplerConfig()).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(langevin.sigma_tm, edm.sigma_tm)
    torch.testing.assert_close(langevin.sigma_t, edm.sigma_t)


@pytest.mark.parametrize(
    "schedule",
    [NoiseScheduleConfig(p=None), NoiseScheduleConfig(p=math.inf)],
    ids=["p_none", "p_inf"],
)
def test_geometric_schedule_is_uniform_in_log_sigma(schedule: NoiseScheduleConfig) -> None:
    """Geometric spacing puts sigma uniformly in log sigma between the schedule bounds."""
    sampler_schedule = AnnealedLangevinSampler(
        LangevinSamplerConfig(schedule=schedule)
    ).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(
        sampler_schedule.sigma_tm[0], torch.tensor(schedule.sigma_data * schedule.s_max)
    )
    torch.testing.assert_close(
        sampler_schedule.sigma_t[-1], torch.tensor(schedule.sigma_data * schedule.s_min)
    )

    expected_ratio = (schedule.s_min / schedule.s_max) ** (1 / NUM_STEPS)
    ratios = sampler_schedule.sigma_t / sampler_schedule.sigma_tm  # (steps,)
    torch.testing.assert_close(ratios, torch.full_like(ratios, expected_ratio))


@pytest.mark.parametrize("p", [7.0, None])
def test_schedule_steps_are_consistent(p: float | None) -> None:
    """Each step starts where the previous one ended, with dt = sigma_t - sigma_tm."""
    config = LangevinSamplerConfig(schedule=NoiseScheduleConfig(p=p))
    schedule = AnnealedLangevinSampler(config).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(schedule.sigma_tm[1:], schedule.sigma_t[:-1])
    torch.testing.assert_close(schedule.dt, schedule.sigma_t - schedule.sigma_tm)
    torch.testing.assert_close(schedule.t_hat, schedule.sigma_tm)
    assert torch.all(schedule.dt < 0)


@pytest.mark.parametrize(
    ("schedule_kwargs", "message"),
    [
        ({"p": 0.0}, "must be positive"),
        ({"p": -1.0}, "must be positive"),
        ({"s_min": 0.0}, "must be positive"),
        ({"s_min": 200.0}, "must be less than s_max"),
        ({"sigma_data": 0.0}, "must be positive"),
    ],
)
def test_schedule_config_rejects_invalid_values(schedule_kwargs: dict, message: str) -> None:
    """Noise levels and a finite p must be positive, with s_min below s_max."""
    with pytest.raises(ValueError, match=message):
        NoiseScheduleConfig(**schedule_kwargs)
