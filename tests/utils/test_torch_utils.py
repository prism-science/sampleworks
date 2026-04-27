"""Tests for PyTorch helpers in sampleworks.utils.torch_utils."""

import random

import numpy as np
import pytest
import torch
from sampleworks.utils.torch_utils import gaussian_prior, seed_all_rngs, seeded_rngs


class TestGaussianPrior:
    """The shared prior behind every flow model wrapper's ``initialize_from_prior``."""

    def test_shape_from_atom_count(self):
        samples = gaussian_prior(4, 25, "cpu")
        assert samples.shape == (4, 25, 3)
        assert samples.device == torch.device("cpu")

    def test_explicit_shape_overrides_atom_count(self):
        samples = gaussian_prior(2, 25, "cpu", shape=(10, 3))
        assert samples.shape == (2, 10, 3)

    def test_explicit_shape_without_atom_count(self):
        samples = gaussian_prior(2, None, "cpu", shape=(10, 3))
        assert samples.shape == (2, 10, 3)

    def test_samples_are_standard_normal(self):
        torch.manual_seed(0)
        samples = gaussian_prior(64, 500, "cpu")
        assert samples.mean().abs() < 0.02
        assert (samples.std() - 1.0).abs() < 0.02

    @pytest.mark.parametrize("shape", [(100,), (100, 4), (2, 100, 3)])
    def test_rejects_shapes_that_are_not_atoms_by_three(self, shape):
        with pytest.raises(ValueError, match="shape"):
            gaussian_prior(2, 25, "cpu", shape=shape)

    def test_requires_atom_count_or_shape(self):
        with pytest.raises(ValueError, match="features or shape"):
            gaussian_prior(2, None, "cpu")


def _draw_from_global_rngs() -> tuple[float, float, torch.Tensor]:
    """One draw from each global generator that the seeding helpers cover."""
    return random.random(), float(np.random.random()), torch.rand(4)


def _assert_same_draws(first, second) -> None:
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


class TestSeedAllRngs:
    """``seed_all_rngs`` makes Python, NumPy and PyTorch global draws reproducible."""

    def test_same_seed_repeats_every_generator(self):
        seed_all_rngs(7)
        first = _draw_from_global_rngs()
        seed_all_rngs(7)
        _assert_same_draws(first, _draw_from_global_rngs())

    def test_different_seeds_change_every_generator(self):
        seed_all_rngs(7)
        first = _draw_from_global_rngs()
        seed_all_rngs(8)
        second = _draw_from_global_rngs()
        assert first[0] != second[0]
        assert first[1] != second[1]
        assert not torch.equal(first[2], second[2])

    def test_seeds_the_gaussian_prior(self):
        seed_all_rngs(3)
        first = gaussian_prior(2, 10, "cpu")
        seed_all_rngs(3)
        assert torch.equal(first, gaussian_prior(2, 10, "cpu"))


class TestSeededRngs:
    """``seeded_rngs`` seeds a block and leaves the surrounding random stream untouched."""

    def test_block_is_reproducible(self):
        with seeded_rngs(11):
            first = _draw_from_global_rngs()
        with seeded_rngs(11):
            _assert_same_draws(first, _draw_from_global_rngs())

    def test_block_matches_seed_all_rngs(self):
        with seeded_rngs(11):
            inside = _draw_from_global_rngs()
        seed_all_rngs(11)
        _assert_same_draws(inside, _draw_from_global_rngs())

    def test_surrounding_stream_is_restored(self):
        seed_all_rngs(5)
        expected = _draw_from_global_rngs()

        seed_all_rngs(5)
        with seeded_rngs(99):
            _draw_from_global_rngs()
        _assert_same_draws(expected, _draw_from_global_rngs())

    def test_state_is_restored_when_the_block_raises(self):
        seed_all_rngs(5)
        expected = _draw_from_global_rngs()

        seed_all_rngs(5)
        with pytest.raises(RuntimeError), seeded_rngs(99):
            _draw_from_global_rngs()
            raise RuntimeError("fails inside the seeded block")
        _assert_same_draws(expected, _draw_from_global_rngs())
