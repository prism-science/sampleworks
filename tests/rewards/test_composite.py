"""Tests for weighted reward combination."""

import pytest
import torch
from sampleworks.core.rewards.composite import CompositeReward
from sampleworks.core.rewards.config import build_reward, RewardConfig
from sampleworks.core.rewards.protocol import RewardFunctionProtocol, RewardInputs
from sampleworks.core.rewards.registry import RewardBuildContext
from sampleworks.utils.guidance_constants import Rewards


class QuadraticReward:
    """Loss = 0.5 * scale * ||coords||^2, so the gradient is scale * coords."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor | None = None,
        b_factors: torch.Tensor | None = None,
        occupancies: torch.Tensor | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return 0.5 * self.scale * (coordinates**2).sum()


class PreparableQuadraticReward(QuadraticReward):
    """A quadratic reward that also binds to the reward inputs."""

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)
        self.prepared_atom_counts: list[int] = []

    def prepare(self, reward_inputs: RewardInputs, *, device="cpu") -> None:
        self.prepared_atom_counts.append(reward_inputs.b_factors.shape[-1])


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
        assert isinstance(CompositeReward([QuadraticReward()]), RewardFunctionProtocol)

    def test_value_is_the_weighted_sum_of_its_terms(self):
        terms = [QuadraticReward(1.0), QuadraticReward(3.0)]
        composite = CompositeReward(terms, [0.25, 0.75])

        combined = composite(coords(), **per_atom())

        expected = 0.25 * terms[0](coords()) + 0.75 * terms[1](coords())
        assert torch.isclose(combined, expected)

    def test_default_weights_average_the_terms(self):
        composite = CompositeReward([QuadraticReward(1.0), QuadraticReward(3.0)])

        assert torch.isclose(composite(coords(), **per_atom()), QuadraticReward(2.0)(coords()))

    def test_gradient_is_the_weighted_sum_of_gradients(self):
        composite = CompositeReward([QuadraticReward(1.0), QuadraticReward(3.0)], [0.5, 0.5])
        x = coords().requires_grad_(True)

        composite(x, **per_atom()).backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, 2.0 * coords())

    def test_a_single_term_is_returned_unweighted(self):
        """A one-term composite must not quietly halve the gradient."""
        composite = CompositeReward([QuadraticReward(2.0)])

        assert torch.isclose(composite(coords(), **per_atom()), QuadraticReward(2.0)(coords()))


class TestCompositeValidation:
    def test_no_rewards_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least one reward function"):
            CompositeReward([])

    def test_mismatched_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="one to one"):
            CompositeReward([QuadraticReward()], [0.5, 0.5])

    def test_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            CompositeReward([QuadraticReward()], [-1.0])


def test_prepare_is_forwarded_only_to_terms_that_need_it():
    n_atoms = 6
    reward_inputs = RewardInputs(
        elements=torch.zeros(1, n_atoms, dtype=torch.long),
        b_factors=torch.full((1, n_atoms), 20.0),
        occupancies=torch.ones(1, n_atoms),
        input_coords=torch.zeros(1, n_atoms, 3),
    )
    preparable = PreparableQuadraticReward()
    composite = CompositeReward([QuadraticReward(), preparable])

    composite.prepare(reward_inputs, device="cpu")

    assert preparable.prepared_atom_counts == [n_atoms]


class TestBuildReward:
    """build_reward turns a configuration into the reward a run scores against."""

    def test_a_single_reward_at_full_weight_is_not_wrapped(self, monkeypatch):
        monkeypatch.setattr(
            "sampleworks.core.rewards.config.build_single_reward",
            lambda reward, options, context: QuadraticReward(),
        )
        config = RewardConfig.single(Rewards.REAL_SPACE_DENSITY, density="m.ccp4", resolution=1.8)

        reward = build_reward(config, RewardBuildContext(structure={}))

        assert isinstance(reward, QuadraticReward)

    def test_several_rewards_are_combined_with_their_weights(self, monkeypatch):
        scales = {Rewards.REAL_SPACE_DENSITY: 1.0, Rewards.STRUCTURE_FACTOR: 3.0}
        monkeypatch.setattr(
            "sampleworks.core.rewards.config.build_single_reward",
            lambda reward, options, context: QuadraticReward(scales[reward]),
        )
        config = RewardConfig.from_mapping(
            {
                "real_space_density": {"weight": 0.25},
                "structure_factor": {"weight": 0.75},
            }
        )

        reward = build_reward(config, RewardBuildContext(structure={}))

        assert isinstance(reward, CompositeReward)
        assert reward.weights == [0.25, 0.75]
        assert torch.isclose(reward(coords(), **per_atom()), QuadraticReward(2.5)(coords()))
