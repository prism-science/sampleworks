"""Tests for :func:`sampleworks.eval.metrics.r_factors`."""

import numpy as np
import pytest
from sampleworks.eval.metrics import r_factors


@pytest.fixture
def amplitudes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``|Fo|``, a noisy ``|Fc|`` and a 10% test set over 1000 reflections."""
    rng = np.random.default_rng(0)
    f_obs = rng.uniform(1.0, 100.0, 1000)
    f_calc = f_obs * rng.uniform(0.8, 1.2, 1000)
    free = np.zeros(1000, dtype=bool)
    free[::10] = True
    return f_obs, f_calc, free


def test_identical_amplitudes_give_zero(amplitudes):
    f_obs, _, free = amplitudes
    r = r_factors(f_obs, f_obs, free)
    assert (r.work, r.free, r.all) == (0.0, 0.0, 0.0)


def test_matches_definition(amplitudes):
    f_obs, f_calc, free = amplitudes
    r = r_factors(f_obs, f_calc, free)

    def r_of(mask):
        return np.abs(f_obs[mask] - f_calc[mask]).sum() / f_obs[mask].sum()

    assert r.work == pytest.approx(r_of(~free))
    assert r.free == pytest.approx(r_of(free))
    assert r.all == pytest.approx(r_of(np.ones_like(free)))


def test_uniform_relative_error_is_that_error():
    """|Fc| = 1.1 |Fo| everywhere is R = 0.1 on every set: no scale is fitted."""
    f_obs = np.linspace(1.0, 50.0, 200)
    free = np.arange(200) % 5 == 0
    r = r_factors(f_obs, 1.1 * f_obs, free)
    assert r.work == pytest.approx(0.1)
    assert r.free == pytest.approx(0.1)
    assert r.all == pytest.approx(0.1)


def test_swapping_the_free_set_changes_r_free_not_r_all(amplitudes):
    f_obs, f_calc, free = amplitudes
    other = np.roll(free, 1)
    r, r_other = r_factors(f_obs, f_calc, free), r_factors(f_obs, f_calc, other)
    assert r.free != r_other.free
    assert r.all == r_other.all


def test_no_test_set_gives_no_r_free(amplitudes):
    f_obs, f_calc, free = amplitudes
    r = r_factors(f_obs, f_calc, np.zeros_like(free))
    assert r.free is None
    assert r.work == pytest.approx(r.all)


def test_rejects_mismatched_lengths(amplitudes):
    f_obs, f_calc, free = amplitudes
    with pytest.raises(ValueError, match="same reflections"):
        r_factors(f_obs, f_calc[:-1], free)


def test_rejects_an_empty_working_set(amplitudes):
    f_obs, f_calc, free = amplitudes
    with pytest.raises(ValueError, match="working set is empty"):
        r_factors(f_obs, f_calc, np.ones_like(free))
