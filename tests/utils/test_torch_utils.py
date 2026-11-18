"""Tests for PyTorch helpers in sampleworks.utils.torch_utils."""

import pytest
import torch
from sampleworks.utils.torch_utils import gaussian_prior


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
