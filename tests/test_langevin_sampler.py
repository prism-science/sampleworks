"""Tests for the Annealed Langevin SDE sampler's schedule and configuration."""

import math

import pytest
import torch
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig


NUM_STEPS = 50


@pytest.mark.parametrize(
    "config",
    [LangevinSamplerConfig(), LangevinSamplerConfig.with_edm_schedule()],
    ids=["default", "with_edm_schedule"],
)
def test_edm_schedule_matches_edm_sampler(config: LangevinSamplerConfig) -> None:
    """The default and with_edm_schedule() Karras schedules are AF3EDMSampler's."""
    langevin = AnnealedLangevinSampler(config).compute_schedule(NUM_STEPS)
    edm = AF3EDMSampler(EDMSamplerConfig()).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(langevin.sigma_tm, edm.sigma_tm)
    torch.testing.assert_close(langevin.sigma_t, edm.sigma_t)


@pytest.mark.parametrize(
    "config",
    [
        LangevinSamplerConfig(p=None),
        LangevinSamplerConfig(p=math.inf),
        LangevinSamplerConfig.with_geometric_schedule(),
    ],
    ids=["p_none", "p_inf", "with_geometric_schedule"],
)
def test_geometric_schedule_is_uniform_in_log_sigma(config: LangevinSamplerConfig) -> None:
    """Geometric spacing puts sigma uniformly in log sigma between the schedule bounds."""
    schedule = AnnealedLangevinSampler(config).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(schedule.sigma_tm[0], torch.tensor(config.sigma_data * config.s_max))
    torch.testing.assert_close(schedule.sigma_t[-1], torch.tensor(config.sigma_data * config.s_min))

    expected_ratio = (config.s_min / config.s_max) ** (1 / NUM_STEPS)
    ratios = schedule.sigma_t / schedule.sigma_tm  # (steps,)
    torch.testing.assert_close(ratios, torch.full_like(ratios, expected_ratio))


@pytest.mark.parametrize("p", [7.0, None])
def test_schedule_steps_are_consistent(p: float | None) -> None:
    """Each step starts where the previous one ended, with dt = sigma_t - sigma_tm."""
    schedule = AnnealedLangevinSampler(LangevinSamplerConfig(p=p)).compute_schedule(NUM_STEPS)

    torch.testing.assert_close(schedule.sigma_tm[1:], schedule.sigma_t[:-1])
    torch.testing.assert_close(schedule.dt, schedule.sigma_t - schedule.sigma_tm)
    torch.testing.assert_close(schedule.t_hat, schedule.sigma_tm)
    assert torch.all(schedule.dt < 0)


def test_schedule_classmethods_apply_overrides() -> None:
    """Overrides replace schedule and non-schedule fields alike."""
    config = LangevinSamplerConfig.with_geometric_schedule(s_min=1e-3, inverse_temperature=4.0)

    assert config.p is None
    assert config.s_min == 1e-3
    assert config.inverse_temperature == 4.0
    assert LangevinSamplerConfig.with_geometric_schedule(p=7.0).p == 7.0


@pytest.mark.parametrize("p", [0.0, -1.0])
def test_config_rejects_nonpositive_p(p: float) -> None:
    """A finite p must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        LangevinSamplerConfig(p=p)


def test_config_rejects_removed_legacy_mode() -> None:
    """The pre-Chroma 'legacy' update is no longer available."""
    with pytest.raises(ValueError, match="sde_mode"):
        LangevinSamplerConfig(sde_mode="legacy")  # ty: ignore[invalid-argument-type]
