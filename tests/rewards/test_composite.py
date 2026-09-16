"""Tests for weighted reward combination."""

import pytest
import torch
from sampleworks.core.rewards.composite import CompositeReward
from sampleworks.core.rewards.config import build_reward, RewardConfig
from sampleworks.core.rewards.protocol import RewardFunctionProtocol, RewardInputs
from sampleworks.utils.guidance_constants import Rewards

from tests.mocks import MockGradientRewardFunction, MockPreparableRewardFunction
from tests.utils.atom_array_builders import build_test_atom_array


def coords(value: float = 2.0) -> torch.Tensor:
    return torch.full((1, 3, 3), value)


def per_atom(n_atoms: int = 3) -> dict:
    return dict(
        elements=torch.ones(1, n_atoms, dtype=torch.long),
        b_factors=torch.full((1, n_atoms), 20.0),
        occupancies=torch.ones(1, n_atoms),
    )


class TestCompositeValue:
    def test_is_a_reward_function(self):
        assert isinstance(CompositeReward([MockGradientRewardFunction()]), RewardFunctionProtocol)

    def test_value_is_the_weighted_sum_of_its_terms(self):
        terms = [MockGradientRewardFunction(1.0), MockGradientRewardFunction(3.0)]
        composite = CompositeReward(terms, [0.25, 0.75])

        combined = composite(coords(), **per_atom())

        expected = 0.25 * terms[0](coords()) + 0.75 * terms[1](coords())
        assert torch.isclose(combined, expected)

    def test_default_weights_average_the_terms(self):
        composite = CompositeReward(
            [MockGradientRewardFunction(1.0), MockGradientRewardFunction(3.0)]
        )

        assert torch.isclose(
            composite(coords(), **per_atom()), MockGradientRewardFunction(2.0)(coords())
        )

    def test_gradient_is_the_weighted_sum_of_gradients(self):
        composite = CompositeReward(
            [MockGradientRewardFunction(1.0), MockGradientRewardFunction(3.0)], [0.5, 0.5]
        )
        x = coords().requires_grad_(True)

        composite(x, **per_atom()).backward()

        assert torch.allclose(x.grad, 2.0 * coords())

    def test_a_single_term_is_returned_unweighted(self):
        """A one-term composite must not quietly halve the gradient."""
        composite = CompositeReward([MockGradientRewardFunction(2.0)])

        assert torch.isclose(
            composite(coords(), **per_atom()), MockGradientRewardFunction(2.0)(coords())
        )


class TestCompositeValidation:
    def test_no_rewards_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least one reward function"):
            CompositeReward([])

    def test_mismatched_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="one to one"):
            CompositeReward([MockGradientRewardFunction()], [0.5, 0.5])

    def test_negative_weight_is_rejected(self):
        """Guards direct construction; configs are checked earlier, in RewardEntry."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CompositeReward([MockGradientRewardFunction()], [-1.0])


def test_prepare_forwards_the_reward_inputs_to_the_terms_that_need_them():
    preparable = MockPreparableRewardFunction()
    composite = CompositeReward([MockGradientRewardFunction(), preparable])
    inputs = RewardInputs.from_atom_array(build_test_atom_array(n_atoms=6), ensemble_size=1)

    composite.prepare(inputs, device="cpu")

    assert preparable.prepared_with == [(6, "cpu")]
    assert preparable.prepared_inputs[0] is inputs


class TestBuildReward:
    """build_reward turns a configuration into the reward a run scores against."""

    def test_a_single_reward_is_not_wrapped_whatever_its_weight(self, monkeypatch):
        monkeypatch.setattr(
            "sampleworks.core.rewards.config.build_single_reward",
            lambda reward, options, device: MockGradientRewardFunction(),
        )
        config = RewardConfig.from_mapping({"real_space_density": {"weight": 0.3}})

        reward = build_reward(config)

        assert isinstance(reward, MockGradientRewardFunction)

    def test_several_rewards_are_combined_with_their_normalized_weights(self, monkeypatch):
        scales = {Rewards.REAL_SPACE_DENSITY: 1.0, Rewards.STRUCTURE_FACTOR: 3.0}
        monkeypatch.setattr(
            "sampleworks.core.rewards.config.build_single_reward",
            lambda reward, options, device: MockGradientRewardFunction(scales[reward]),
        )
        config = RewardConfig.from_mapping(
            {"real_space_density": {"weight": 1.0}, "structure_factor": {"weight": 3.0}}
        )

        reward = build_reward(config)

        assert isinstance(reward, CompositeReward)
        assert reward.weights == [0.25, 0.75]
        assert torch.isclose(
            reward(coords(), **per_atom()), MockGradientRewardFunction(2.5)(coords())
        )
