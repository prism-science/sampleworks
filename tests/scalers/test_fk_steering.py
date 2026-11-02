"""Tests for FK steering resampling and output options."""

import math

import pytest
import torch
from sampleworks.core.samplers.edm import AF3EDMSampler
from sampleworks.core.scalers.fk_steering import FKSteering, resample_indices
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler

from tests.mocks.losses import MockGradientLossFunction
from tests.mocks.model_wrappers import MockFlowModelWrapper


class TestResampleIndices:
    def test_systematic_gives_floor_or_ceil_offspring(self):
        weights = torch.tensor([0.5, 0.25, 0.25, 0.0])
        for _ in range(20):
            indices = resample_indices(weights.log(), "systematic")
            counts = torch.bincount(indices, minlength=4)
            assert counts.tolist() == [2, 1, 1, 0]

    @pytest.mark.parametrize("method", ["multinomial", "systematic"])
    def test_groups_resample_independently(self, method):
        log_weights = torch.randn(12)
        for _ in range(20):
            indices = resample_indices(log_weights, method, group_size=4)
            torch.testing.assert_close(indices // 4, torch.arange(12) // 4)

    def test_last_group_may_be_smaller(self):
        indices = resample_indices(torch.zeros(5), "systematic", group_size=2)
        assert indices.shape == (5,)
        assert indices[4] == 4

    @pytest.mark.parametrize("method", ["multinomial", "systematic"])
    def test_non_finite_particles_lose(self, method):
        log_weights = torch.tensor([math.nan, 0.0, math.inf, 0.0])
        for _ in range(20):
            indices = resample_indices(log_weights, method)
            assert set(indices.tolist()) <= {1, 3}

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="resampling method"):
            resample_indices(torch.zeros(3), "residual")  # ty:ignore[invalid-argument-type]


class TestFKSteeringOutput:
    def _sample(self, steering: FKSteering, structure: dict, edm_sampler: AF3EDMSampler):
        return steering.sample(
            structure=structure,
            model=MockFlowModelWrapper(num_atoms=50),
            sampler=edm_sampler,
            step_scaler=DataSpaceDPSScaler(step_size=0.0),
            loss_function=MockGradientLossFunction(),
            num_particles=3,
        )

    def test_default_returns_lowest_loss_particle(self, mock_structure, edm_sampler):
        out = self._sample(FKSteering(ensemble_size=2, num_steps=4), mock_structure, edm_sampler)
        assert torch.as_tensor(out.final_state).shape == (2, 50, 3)

    @pytest.mark.parametrize("method", ["multinomial", "systematic"])
    def test_return_all_particles(self, mock_structure, edm_sampler, method):
        steering = FKSteering(
            ensemble_size=2,
            num_steps=4,
            resampling_interval=2,
            resampling_method=method,
            resampling_group_size=2,
            return_all_particles=True,
        )
        final = torch.as_tensor(self._sample(steering, mock_structure, edm_sampler).final_state)
        assert final.shape == (6, 50, 3)
        assert torch.isfinite(final).all()


def test_weights_telescope_from_the_previous_resample():
    """With resampling_interval > 1, the weight covers every step since the last resample."""
    steering = FKSteering(fk_lambda=100.0, resampling_method="systematic")
    coords = torch.tensor([0.0, 1.0]).reshape(2, 1, 1).expand(2, 1, 3)  # particle 0 and 1
    # Particle 0's loss fell from 10 to 5 since the last resample, then held for a step.
    history = [torch.tensor([10.0, 0.0]), torch.tensor([5.0, 0.0]), torch.tensor([5.0, 0.0])]

    resampled, loss_at_resample = steering._resample_particles(
        coords=coords,
        loss_curr=history[-1],
        loss_at_last_resample=history[0],
        loss_history=history,
        log_proposal_correction=None,
        num_particles=2,
    )
    assert resampled[:, 0, 0].tolist() == [0.0, 0.0]
    assert loss_at_resample.tolist() == [5.0, 5.0]
