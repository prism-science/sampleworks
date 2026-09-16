"""Comprehensive integration tests for the sampling pipeline.

This file consolidates tests for:
- All wrapper x sampler combinations (parametrized)
- All step scalers with proper reward context (not mock-only)
- All trajectory scalers with real implementations
- Guidance effectiveness ground truth tests
- Numerical stability throughout trajectories
- Device consistency across all wrappers
- Partial diffusion behavior
- EDM schedule behavior

Tests are organized from fast (mock-based) to slow (real wrappers).
"""

import pytest
import torch
from sampleworks.core.rewards.protocol import RewardInputs
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.protocol import SamplerStepOutput, StepParams
from sampleworks.core.scalers.protocol import GuidanceOutput
from sampleworks.core.scalers.step_scalers import (
    DataSpaceDPSScaler,
    NoiseSpaceDPSScaler,
    NoScalingScaler,
)
from sampleworks.utils.guidance_constants import (
    StepScalers,
    StructurePredictor,
    TrajectorySamplers,
    TrajectoryScalers,
)
from sampleworks.utils.structure_utils import process_structure_to_trajectory_input
from torch import Tensor

from tests.conftest import (
    annotate_structure_for_wrapper_type,
    create_sampler_from_type,
    create_step_scaler_from_type,
    create_trajectory_scaler_from_type,
    get_all_step_scalers,
    get_all_trajectory_samplers,
    get_all_trajectory_scalers,
    get_fixture_name_for_wrapper_type,
    get_slow_wrappers,
    STEP_SCALER_REGISTRY,
    STRUCTURES,
)
from tests.mocks import (
    MismatchCase,
    MismatchCaseWrapper,
    MockFlowModelWrapper,
    MockPreparableRewardFunction,
    MockStepScaler,
)
from tests.mocks.rewards import MockGradientRewardFunction
from tests.utils.atom_array_builders import build_test_atom_array


def create_step_context_with_reward(
    step_index: int,
    state: Tensor,
    device: torch.device,
    *,
    total_steps: int = 10,
    t: float = 0.5,
    dt: float = -0.1,
    noise_scale: float = 0.1,
    include_x_t: bool = False,
) -> StepParams:
    """Create StepParams with reward for testing DPS scalers.

    Parameters
    ----------
    step_index
        Current step index.
    state
        Current state tensor, shape (batch, atoms, 3).
    device
        Device to place tensors on.
    total_steps
        Total number of steps in the trajectory.
    t
        Current diffusion time.
    dt
        Step size in time (negative for reverse diffusion).
    noise_scale
        Noise scale for the step.
    include_x_t
        If True, include x_t in metadata for NoiseSpaceDPSScaler.

    Returns
    -------
    StepParams
        Context with reward and reward_inputs populated.
    """
    batch_size, num_atoms, _ = state.shape
    reward = MockGradientRewardFunction()
    reward_inputs = RewardInputs(
        elements=torch.ones(batch_size, num_atoms, device=device),
        b_factors=torch.ones(batch_size, num_atoms, device=device) * 20.0,
        occupancies=torch.ones(batch_size, num_atoms, device=device),
        input_coords=state.clone(),
    )

    metadata = None
    if include_x_t:
        x_t = state.clone().detach().requires_grad_(True)
        metadata = {"x_t": x_t}

    return StepParams(
        step_index=step_index,
        total_steps=total_steps,
        t=torch.tensor([t] * batch_size, device=device),
        dt=torch.tensor([dt] * batch_size, device=device),
        noise_scale=torch.tensor([noise_scale] * batch_size, device=device),
        reward=reward,
        reward_inputs=reward_inputs,
        metadata=metadata,
    )


# ============================================================================
# Fast mock-based tests (no checkpoints required)
# ============================================================================


@pytest.mark.parametrize("sampler_type", get_all_trajectory_samplers(), ids=lambda s: s.value)
class TestWrapperSamplerMatrixMock:
    """Test wrapper and sampler combinations with mock components."""

    def test_single_step_produces_valid_output(
        self,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
    ):
        """Single step produces valid output with correct shape and device."""
        sampler = create_sampler_from_type(sampler_type, device=device)

        features = mock_wrapper.featurize(mock_structure)
        state = mock_wrapper.initialize_from_prior(batch_size=2, features=features)

        schedule = sampler.compute_schedule(num_steps=5)
        context = sampler.get_context_for_step(0, schedule)

        step_output = sampler.step(state, mock_wrapper, context, features=features)

        assert step_output.state.shape == state.shape
        assert step_output.state.device == state.device
        assert torch.isfinite(step_output.state).all()
        assert isinstance(step_output, SamplerStepOutput)

    def test_full_trajectory_preserves_shape(
        self,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
    ):
        """Full trajectory preserves shape throughout all steps."""
        sampler = create_sampler_from_type(sampler_type, device=device)

        features = mock_wrapper.featurize(mock_structure)
        batch_size = 2
        state = mock_wrapper.initialize_from_prior(batch_size=batch_size, features=features)
        initial_shape = state.shape

        num_steps = 5
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state, mock_wrapper, context, features=features)
            state = step_output.state
            assert state.shape == initial_shape, f"Shape changed at step {i}"

        assert state.shape == (batch_size, mock_wrapper.num_atoms, 3)

    def test_step_index_monotonic_in_context(
        self,
        sampler_type: TrajectorySamplers,
        device: torch.device,
    ):
        """Step contexts have monotonically increasing step_index."""
        sampler = create_sampler_from_type(sampler_type, device=device)
        num_steps = 10
        schedule = sampler.compute_schedule(num_steps=num_steps)

        indices = []
        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            indices.append(context.step_index)

        assert indices == list(range(num_steps))


class TestStepScalerMatrix:
    """Test step scalers with mock wrapper and ground truth assertions."""

    def test_no_scaling_returns_zeros(self, device: torch.device):
        """NoScalingScaler returns zero guidance and zero loss."""
        scaler = NoScalingScaler()
        batch_size = 2
        num_atoms = 10
        state = torch.randn(batch_size, num_atoms, 3, device=device)

        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5] * batch_size, device=device),
            dt=torch.tensor([-0.1] * batch_size, device=device),
        )

        direction, loss = scaler.scale(state, context, model=None)

        # Ground truth: NoScalingScaler always returns exact zeros
        assert torch.allclose(direction, torch.zeros_like(state))
        assert torch.allclose(loss, torch.zeros(batch_size, device=device))
        assert direction.shape == state.shape
        assert loss.shape == (batch_size,)

    def test_no_scaling_guidance_strength_returns_zeros(self, device: torch.device):
        """NoScalingScaler.guidance_strength returns zero tensor."""
        scaler = NoScalingScaler()
        batch_size = 2

        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5] * batch_size, device=device),
            dt=torch.tensor([-0.1] * batch_size, device=device),
        )

        strength = scaler.guidance_strength(context)
        assert torch.allclose(strength, torch.zeros(batch_size, device=device))

    def test_data_space_dps_gradient_equals_state(self, device: torch.device):
        """DataSpaceDPSScaler gradient equals state when using MockGradientRewardFunction.

        MockGradientRewardFunction computes loss = 0.5 * ||coords||^2,
        so grad(loss) w.r.t. coords = coords.
        """
        scaler = DataSpaceDPSScaler(step_size=0.1)
        batch_size = 2
        num_atoms = 10

        # Use fixed seed for reproducibility
        torch.manual_seed(123)
        state = torch.randn(batch_size, num_atoms, 3, device=device)

        context = create_step_context_with_reward(0, state, device)

        direction, loss = scaler.scale(state, context, model=None)

        # Ground truth: direction = grad(0.5 * ||state||^2) = state
        assert torch.allclose(direction, state, atol=1e-6)
        assert torch.isfinite(direction).all()
        assert torch.isfinite(loss).all()

        # Ground truth: loss = 0.5 * ||state||^2
        expected_loss = 0.5 * (state**2).sum()
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_data_space_dps_guidance_strength_equals_step_size(self, device: torch.device):
        """DataSpaceDPSScaler.guidance_strength returns step_size for all batch elements."""
        step_size = 0.15
        scaler = DataSpaceDPSScaler(step_size=step_size)
        batch_size = 3

        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5] * batch_size, device=device),
            dt=torch.tensor([-0.1] * batch_size, device=device),
        )

        strength = scaler.guidance_strength(context)

        # Ground truth: strength = step_size for all batch elements
        expected = torch.tensor([step_size] * batch_size, device=device)
        assert torch.allclose(strength, expected)

    def test_data_space_dps_requires_reward_context(self, device: torch.device):
        """DataSpaceDPSScaler raises error without reward context."""
        scaler = DataSpaceDPSScaler(step_size=0.1)
        state = torch.randn(2, 10, 3, device=device)

        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5, 0.5], device=device),
            dt=torch.tensor([-0.1, -0.1], device=device),
        )

        with pytest.raises(ValueError, match="missing reward"):
            scaler.scale(state, context, model=None)

    def test_noise_space_dps_requires_x_t_in_metadata(self, device: torch.device):
        """NoiseSpaceDPSScaler raises error without x_t in metadata."""
        scaler = NoiseSpaceDPSScaler(step_size=0.1)
        batch_size = 2
        num_atoms = 10
        state = torch.randn(batch_size, num_atoms, 3, device=device)

        context = create_step_context_with_reward(0, state, device, include_x_t=False)

        with pytest.raises(ValueError, match="x_t"):
            scaler.scale(state, context, model=None)

    def test_noise_space_dps_gradient_chain_rule(self, device: torch.device):
        """NoiseSpaceDPSScaler gradient follows chain rule through model.

        With state = x_t * 0.9 + 0.1, and loss = 0.5 * ||state||^2,
        grad(loss) w.r.t. x_t = state * 0.9 (chain rule).
        """
        scaler = NoiseSpaceDPSScaler(step_size=0.1)
        batch_size = 2
        num_atoms = 10

        torch.manual_seed(456)
        x_t = torch.randn(batch_size, num_atoms, 3, device=device, requires_grad=True)
        state = x_t * 0.9 + 0.1

        reward = MockGradientRewardFunction()
        reward_inputs = RewardInputs(
            elements=torch.ones(batch_size, num_atoms, device=device),
            b_factors=torch.ones(batch_size, num_atoms, device=device) * 20.0,
            occupancies=torch.ones(batch_size, num_atoms, device=device),
            input_coords=state.detach().clone(),
        )
        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5] * batch_size, device=device),
            dt=torch.tensor([-0.1] * batch_size, device=device),
            noise_scale=torch.tensor([0.1] * batch_size, device=device),
            reward=reward,
            reward_inputs=reward_inputs,
            metadata={"x_t": x_t},
        )

        direction, loss = scaler.scale(state, context, model=None)

        # Ground truth: direction = d(loss)/d(x_t) = d(loss)/d(state) * d(state)/d(x_t)
        #             = state * 0.9
        assert torch.allclose(direction, state * 0.9, atol=1e-6)
        assert torch.isfinite(direction).all()
        assert torch.isfinite(loss).all()

    @pytest.mark.parametrize("scaler_type", get_all_step_scalers(), ids=lambda s: s.value)
    def test_scaler_scale_returns_correct_shapes(
        self,
        scaler_type: StepScalers,
        device: torch.device,
    ):
        """All step scalers return (direction, loss) with correct shapes."""
        info = STEP_SCALER_REGISTRY[scaler_type]
        scaler = create_step_scaler_from_type(scaler_type)

        batch_size = 2
        num_atoms = 10

        if scaler_type == StepScalers.NOISE_SPACE_DPS:
            x_t = torch.randn(batch_size, num_atoms, 3, device=device, requires_grad=True)
            state = x_t * 0.9 + 0.1
            reward = MockGradientRewardFunction()
            reward_inputs = RewardInputs(
                elements=torch.ones(batch_size, num_atoms, device=device),
                b_factors=torch.ones(batch_size, num_atoms, device=device) * 20.0,
                occupancies=torch.ones(batch_size, num_atoms, device=device),
                input_coords=state.detach().clone(),
            )
            context = StepParams(
                step_index=0,
                total_steps=10,
                t=torch.tensor([0.5] * batch_size, device=device),
                dt=torch.tensor([-0.1] * batch_size, device=device),
                noise_scale=torch.tensor([0.1] * batch_size, device=device),
                reward=reward,
                reward_inputs=reward_inputs,
                metadata={"x_t": x_t},
            )
        else:
            state = torch.randn(batch_size, num_atoms, 3, device=device)
            if info.requires_reward:
                context = create_step_context_with_reward(0, state, device)
            else:
                context = StepParams(
                    step_index=0,
                    total_steps=10,
                    t=torch.tensor([0.5] * batch_size, device=device),
                    dt=torch.tensor([-0.1] * batch_size, device=device),
                )

        direction, loss = scaler.scale(state, context, model=None)

        assert direction.shape == state.shape
        assert loss.ndim <= 1

    @pytest.mark.parametrize("scaler_type", get_all_step_scalers(), ids=lambda s: s.value)
    def test_scaler_guidance_strength_returns_correct_shape(
        self,
        scaler_type: StepScalers,
        device: torch.device,
    ):
        """All step scalers return guidance_strength with correct shape."""
        scaler = create_step_scaler_from_type(scaler_type)
        batch_size = 3

        context = StepParams(
            step_index=0,
            total_steps=10,
            t=torch.tensor([0.5] * batch_size, device=device),
            dt=torch.tensor([-0.1] * batch_size, device=device),
        )

        strength = scaler.guidance_strength(context)
        assert strength.shape == (batch_size,)


class TestMockWrapperGroundTruth:
    """Ground truth tests validating MockFlowModelWrapper behavior."""

    def test_mock_wrapper_step_returns_target_when_set(self, device: torch.device):
        """MockFlowModelWrapper.step() returns target when target is provided."""
        batch_size = 2
        num_atoms = 20
        target = torch.ones(batch_size, num_atoms, 3, device=device) * 5.0
        wrapper = MockFlowModelWrapper(num_atoms=num_atoms, device=device, target=target)

        features = wrapper.featurize({})
        state = torch.randn(batch_size, num_atoms, 3, device=device)
        t = torch.tensor([0.5] * batch_size, device=device)

        result = wrapper.step(state, t, features=features)

        # Ground truth: step() returns target exactly
        assert torch.allclose(result, target)

    def test_mock_wrapper_step_returns_zeros_without_target(self, device: torch.device):
        """MockFlowModelWrapper.step() returns zeros when no target is set."""
        batch_size = 2
        num_atoms = 20
        wrapper = MockFlowModelWrapper(num_atoms=num_atoms, device=device)

        features = wrapper.featurize({})
        state = torch.randn(batch_size, num_atoms, 3, device=device)
        t = torch.tensor([0.5] * batch_size, device=device)

        result = wrapper.step(state, t, features=features)

        # Ground truth: step() returns zeros when no target
        assert torch.allclose(result, torch.zeros_like(state))

    def test_trajectory_converges_to_target(self, device: torch.device):
        """Full trajectory should converge toward target with MockFlowModelWrapper."""
        batch_size = 1
        num_atoms = 20
        target = torch.ones(batch_size, num_atoms, 3, device=device) * 2.0
        wrapper = MockFlowModelWrapper(num_atoms=num_atoms, device=device, target=target)
        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )

        features = wrapper.featurize({})
        torch.manual_seed(789)
        state = wrapper.initialize_from_prior(batch_size=batch_size, features=features)
        initial_distance = (state - target).norm()

        num_steps = 10
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state, wrapper, context, features=features)
            state = step_output.state

        final_distance = (state - target).norm()

        # Ground truth: final state should be closer to target than initial
        assert final_distance < initial_distance

    def test_trajectory_converges_to_zero_without_target(self, device: torch.device):
        """Full trajectory should converge toward zero without target."""
        batch_size = 1
        num_atoms = 20
        wrapper = MockFlowModelWrapper(num_atoms=num_atoms, device=device)
        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )

        features = wrapper.featurize({})
        torch.manual_seed(101)
        state = wrapper.initialize_from_prior(batch_size=batch_size, features=features)
        initial_norm = state.norm()

        num_steps = 10
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state, wrapper, context, features=features)
            state = step_output.state

        final_norm = state.norm()

        # Ground truth: final state should have smaller norm (converging to zero)
        assert final_norm < initial_norm


class TestTrajectoryScalerMatrixMock:
    """Test trajectory scaler combinations with mock components."""

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    @pytest.mark.parametrize("sampler_type", get_all_trajectory_samplers(), ids=lambda s: s.value)
    def test_trajectory_scaler_initializes_from_prior_and_returns_guidance_output(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
        mock_gradient_reward: MockGradientRewardFunction,
    ):
        """TrajectoryScaler.sample() initializes from the prior and returns valid output."""
        sampler = create_sampler_from_type(sampler_type, device=device)
        num_steps = 5
        ensemble_size = 2

        trajectory_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            ensemble_size=ensemble_size,
            num_steps=num_steps,
        )

        result = trajectory_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=1,
        )

        assert isinstance(result, GuidanceOutput)
        assert result.final_state.shape == (ensemble_size, mock_wrapper.num_atoms, 3)
        assert result.structure is mock_structure
        assert result.trajectory is not None
        assert len(result.trajectory) == num_steps
        assert torch.isfinite(torch.as_tensor(result.final_state)).all()
        assert len(mock_wrapper.prior_initialization_features) == 1
        assert mock_wrapper.prior_initialization_features[0] is not None
        assert not hasattr(mock_wrapper.prior_initialization_features[0], "x_init")

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    def test_trajectory_scaler_handles_multiple_particles(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
        mock_gradient_reward: MockGradientRewardFunction,
    ):
        """TrajectoryScaler handles multiple particles correctly."""
        sampler = create_sampler_from_type(TrajectorySamplers.AF3EDM, device=device)
        num_particles = 3

        trajectory_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            ensemble_size=1,
            num_steps=3,
        )

        result = trajectory_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=num_particles,
        )

        assert isinstance(result, GuidanceOutput)
        assert result.final_state is not None
        assert torch.isfinite(torch.as_tensor(result.final_state)).all()

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    def test_preparable_reward_is_prepared_with_model_topology_before_first_call(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        device: torch.device,
    ):
        """Two-phase rewards are prepared with model-topology reward inputs before any evaluation.

        The model and the structure deliberately have different atom counts. A reward
        prepared against the input structure would bind the wrong atom ordering, which
        is how the structure-factor reward silently scores the wrong thing. The inputs
        handed to ``prepare`` carry the model atom array as their topology.
        """

        struct_atom_array = build_test_atom_array(
            chain_ids=["A"] * 5,
            res_ids=[1, 2, 3, 4, 5],
            atom_names=["N", "CA", "C", "O", "CB"],
        )
        case = MismatchCase(
            id="model_drops_one_atom",
            description="Model represents four of the structure's five atoms.",
            model_atom_array=struct_atom_array[:4].copy(),
            struct_atom_array=struct_atom_array,
            expected_n_common=4,
            expected_has_mismatch=True,
        )
        wrapper = MismatchCaseWrapper(case, device=device)
        structure = {"asym_unit": struct_atom_array, "metadata": {"id": "mismatch"}}

        reward = MockPreparableRewardFunction()
        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )
        trajectory_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            ensemble_size=1,
            num_steps=3,
        )

        trajectory_scaler.sample(
            structure=structure,
            model=wrapper,
            sampler=sampler,
            # A real step scaler, so the reward is actually evaluated and the
            # before-first-call assertion below has something to be true about.
            step_scaler=DataSpaceDPSScaler(step_size=0.1),
            reward=reward,
            num_particles=1,
        )

        assert reward.prepared_atom_counts == [case.n_model]
        (prepared,) = reward.prepared_inputs
        assert prepared.atom_array is not None
        assert prepared.atom_array.array_length() == case.n_model
        assert list(prepared.atom_array.atom_name) == list(case.model_atom_array.atom_name)
        assert reward.calls > 0
        assert reward.calls_before_prepare == 0


class TestPartialDiffusion:
    """Test partial diffusion (t_start > 0) behavior."""

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    @pytest.mark.parametrize("sampler_type", get_all_trajectory_samplers(), ids=lambda s: s.value)
    def test_t_start_reduces_trajectory_length(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_gradient_reward: MockGradientRewardFunction,
        mock_step_scaler: MockStepScaler,
    ):
        """t_start=0.5 should produce trajectory of length num_steps/2."""
        sampler = create_sampler_from_type(sampler_type, device=device)
        num_steps = 10
        scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            t_start=0.5,
            num_steps=num_steps,
            ensemble_size=1,
        )

        result = scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=1,
        )

        expected_length = num_steps - int(0.5 * num_steps)
        assert len(result.trajectory) == expected_length

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    @pytest.mark.parametrize("sampler_type", get_all_trajectory_samplers(), ids=lambda s: s.value)
    def test_full_trajectory_longer_than_partial(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_gradient_reward: MockGradientRewardFunction,
        mock_step_scaler: MockStepScaler,
    ):
        """Full trajectory (t_start=0) should be longer than partial (t_start=0.5)."""
        sampler = create_sampler_from_type(sampler_type, device=device)
        num_steps = 10

        full_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            t_start=0.0,
            num_steps=num_steps,
            ensemble_size=1,
        )
        partial_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            t_start=0.5,
            num_steps=num_steps,
            ensemble_size=1,
        )

        full_result = full_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=1,
        )
        partial_result = partial_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=1,
        )

        assert len(full_result.trajectory) > len(partial_result.trajectory)


class TestGuidanceEffectivenessMock:
    """Guidance should produce valid output with mock components."""

    @pytest.mark.parametrize(
        "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
    )
    def test_trajectory_scaler_guidance_completes(
        self,
        trajectory_scaler_type: TrajectoryScalers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
        mock_gradient_reward: MockGradientRewardFunction,
    ):
        """All trajectory scalers complete with valid output."""
        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )

        trajectory_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            ensemble_size=1,
            num_steps=5,
        )

        result = trajectory_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=2,
        )

        assert isinstance(result, GuidanceOutput)
        assert result.final_state.shape == (1, mock_wrapper.num_atoms, 3)
        assert torch.isfinite(torch.as_tensor(result.final_state)).all()
        assert result.trajectory is not None
        assert len(result.trajectory) == 5


class TestNumericalStabilityMock:
    """Verify numerical stability with mock components."""

    def test_extended_trajectory_produces_no_nans(
        self,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
    ):
        """Extended trajectory (200 steps) produces no NaN values."""
        sampler = create_sampler_from_type(TrajectorySamplers.AF3EDM, device=device)

        features = mock_wrapper.featurize(mock_structure)
        state = mock_wrapper.initialize_from_prior(batch_size=1, features=features)

        num_steps = 200
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(
                state, mock_wrapper, context, scaler=mock_step_scaler, features=features
            )

            assert not torch.isnan(step_output.state).any(), f"NaN detected at step {i}"
            assert not torch.isinf(step_output.state).any(), f"Inf detected at step {i}"

            state = step_output.state

    def test_guidance_does_not_introduce_nans(
        self,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
        mock_gradient_reward: MockGradientRewardFunction,
    ):
        """Guidance application produces finite values throughout."""
        from sampleworks.core.scalers.pure_guidance import PureGuidance

        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )

        trajectory_scaler = PureGuidance(
            ensemble_size=2,
            num_steps=20,
        )

        result = trajectory_scaler.sample(
            structure=mock_structure,
            model=mock_wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
        )

        assert torch.isfinite(torch.as_tensor(result.final_state)).all()

        assert result.trajectory is not None
        for i, traj_state in enumerate(result.trajectory):
            assert torch.isfinite(torch.as_tensor(traj_state)).all(), (
                f"Step {i} state has non-finite values"
            )


class TestDeviceConsistencyMock:
    """Verify device placement is consistent with mock components."""

    @pytest.mark.parametrize("sampler_type", get_all_trajectory_samplers(), ids=lambda s: s.value)
    def test_tensors_on_same_device_throughout_trajectory(
        self,
        sampler_type: TrajectorySamplers,
        device: torch.device,
        mock_wrapper: MockFlowModelWrapper,
        mock_structure: dict,
        mock_step_scaler: MockStepScaler,
    ):
        """All tensors remain on the same device throughout sampling."""
        sampler = create_sampler_from_type(sampler_type, device=device)

        features = mock_wrapper.featurize(mock_structure)
        state = mock_wrapper.initialize_from_prior(batch_size=1, features=features)

        assert state.device == device

        num_steps = 5
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(
                state, mock_wrapper, context, scaler=mock_step_scaler, features=features
            )

            assert step_output.state.device == device, (
                f"State moved to {step_output.state.device} at step {i}"
            )
            assert step_output.denoised.device == device
            assert step_output.loss.device == device

            state = step_output.state


class TestEDMScheduleBehavior:
    """EDM schedule computation tests (sampler-specific, no wrapper needed)."""

    def test_schedule_sigma_decreases(self, edm_sampler: AF3EDMSampler):
        """Sigma values should decrease over the trajectory."""
        schedule = edm_sampler.compute_schedule(num_steps=20)

        sigma_values = schedule.sigma_tm.tolist()
        for i in range(len(sigma_values) - 1):
            assert sigma_values[i] > sigma_values[i + 1]

    def test_schedule_length_matches_num_steps(self, edm_sampler: AF3EDMSampler):
        """Schedule arrays should match num_steps length."""
        num_steps = 15
        schedule = edm_sampler.compute_schedule(num_steps=num_steps)

        assert len(schedule.sigma_tm) == num_steps
        assert len(schedule.sigma_t) == num_steps
        assert len(schedule.gamma) == num_steps

    def test_context_dt_is_negative(self, edm_sampler: AF3EDMSampler):
        """dt should be negative (time flows backward in reverse diffusion)."""
        schedule = edm_sampler.compute_schedule(num_steps=10)

        for i in range(10):
            context = edm_sampler.get_context_for_step(i, schedule)
            assert context.dt is not None and context.dt < 0, f"dt should be negative at step {i}"


class TestOutputShapeConsistency:
    """Verify output shapes are consistent across batch sizes and atom counts."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4], ids=lambda b: f"batch_{b}")
    @pytest.mark.parametrize("num_atoms", [10, 50, 100], ids=lambda n: f"atoms_{n}")
    def test_shape_invariants(
        self,
        batch_size: int,
        num_atoms: int,
        device: torch.device,
    ):
        """Output shapes match (batch_size, num_atoms, 3) throughout."""
        wrapper = MockFlowModelWrapper(num_atoms=num_atoms, device=device)
        sampler = create_sampler_from_type(TrajectorySamplers.AF3EDM, device=device)

        features = wrapper.featurize({})
        state = wrapper.initialize_from_prior(batch_size=batch_size, features=features)

        assert state.shape == (batch_size, num_atoms, 3)

        schedule = sampler.compute_schedule(num_steps=3)
        context = sampler.get_context_for_step(0, schedule)
        step_output = sampler.step(state, wrapper, context, features=features)

        assert step_output.state.shape == (batch_size, num_atoms, 3)
        assert step_output.denoised.shape == (batch_size, num_atoms, 3)


class TestSamplingDeterminism:
    """Tests for determinism in sampling."""

    def test_sampling_determinism_with_seed(self, device: torch.device):
        """Sampling should be deterministic with fixed seed."""
        wrapper = MockFlowModelWrapper(num_atoms=20, device=device)
        sampler = AF3EDMSampler(EDMSamplerConfig(device=device))

        features = wrapper.featurize({})

        torch.manual_seed(42)
        state1 = wrapper.initialize_from_prior(batch_size=1, features=features)
        schedule = sampler.compute_schedule(num_steps=3)
        for i in range(3):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state1, wrapper, context, features=features)
            state1 = step_output.state

        torch.manual_seed(42)
        state2 = wrapper.initialize_from_prior(batch_size=1, features=features)
        schedule = sampler.compute_schedule(num_steps=3)
        for i in range(3):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state2, wrapper, context, features=features)
            state2 = step_output.state

        assert torch.allclose(state1, state2)


# ============================================================================
# Slow tests with real model wrappers (require checkpoints)
# All tests parametrized across ALL wrappers to avoid duplicates.
# ============================================================================


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", get_slow_wrappers(), ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestRealWrapperSamplerMatrix:
    """Parametrized integration tests with real model wrappers.

    All wrappers are tested with all structures.
    """

    def test_featurize_produces_valid_input(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """featurize() should produce valid conditioning for prior initialization."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        assert features is not None
        assert features.conditioning is not None
        assert not hasattr(features, "x_init")

    def test_single_step_runs(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """Single step should execute without errors."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )
        schedule = sampler.compute_schedule(num_steps=10)
        context = sampler.get_context_for_step(0, schedule)

        step_output = sampler.step(
            state=state,
            model_wrapper=wrapper,
            context=context,
            features=features,
        )

        assert step_output.state.shape == state.shape
        assert torch.isfinite(step_output.state).all()

    def test_full_trajectory_completes(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """Full trajectory should complete without errors and finite values."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )
        num_steps = 5
        schedule = sampler.compute_schedule(num_steps=num_steps)

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(
                state=state,
                model_wrapper=wrapper,
                context=context,
                features=features,
            )
            state = step_output.state
            assert torch.isfinite(state).all(), f"Non-finite values at step {i}"

        assert state.shape[0] == 1

    def test_device_consistency(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """Wrapper maintains device consistency throughout sampling."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        assert state.device == device

        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )
        schedule = sampler.compute_schedule(num_steps=3)
        context = sampler.get_context_for_step(0, schedule)

        step_output = sampler.step(state, wrapper, context, features=features)

        assert step_output.state.device == device


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", get_slow_wrappers(), ids=lambda w: w.value)
@pytest.mark.parametrize(
    "trajectory_scaler_type", get_all_trajectory_scalers(), ids=lambda s: s.value
)
class TestRealTrajectoryScalerMatrix:
    """Test trajectory scalers with all real wrappers."""

    def test_trajectory_scaler_returns_guidance_output(
        self,
        wrapper_type: StructurePredictor,
        trajectory_scaler_type: TrajectoryScalers,
        temp_output_dir,
        request,
        mock_step_scaler: MockStepScaler,
        mock_gradient_reward: MockGradientRewardFunction,
    ):
        """TrajectoryScaler.sample() returns valid GuidanceOutput with real wrapper."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue("structure_1vme")
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        sampler = create_sampler_from_type(TrajectorySamplers.AF3EDM, device=device)

        trajectory_scaler = create_trajectory_scaler_from_type(
            trajectory_scaler_type,
            ensemble_size=1,
            num_steps=3,
        )

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)

        result = trajectory_scaler.sample(
            structure=annotated,
            model=wrapper,
            sampler=sampler,
            step_scaler=mock_step_scaler,
            reward=mock_gradient_reward,
            num_particles=1,
        )

        assert isinstance(result, GuidanceOutput)
        assert result.final_state is not None
        assert torch.isfinite(torch.as_tensor(result.final_state)).all()
        assert result.trajectory is not None
        assert len(result.trajectory) == 3


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", get_slow_wrappers(), ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestRealWrapperPreprocessing:
    """process_structure_to_trajectory_input feeds properly into to_reward_inputs with real wrappers

    Verifies that featurization and preprocessing produce reward_inputs whose
    atom dimension matches the state dimension for every wrapper by structure
    combination.  This catches model-specific NaN coordinate or occupancy
    issues (e.g., RF3 on 5I09) that only surface with real featurization.
    """

    def test_reward_inputs_are_valid(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """reward_inputs atom count equals the model state atom count.

        Wrappers must ensure model_atom_array has valid coordinates and
        occupancy for all atoms.  ``RewardInputs.from_atom_array`` uses
        all atoms without masking, so any NaN or zero-occupancy atoms
        would produce invalid reward tensors and a dimension mismatch
        in the step scalers.
        """
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        processed = process_structure_to_trajectory_input(
            structure=annotated,
            coords_from_prior=state,
            features=features,
            ensemble_size=1,
        )
        reward_inputs = processed.to_reward_inputs(device=device)

        n_state = state.shape[-2]
        n_reward = reward_inputs.elements.shape[-1]
        assert n_state == n_reward, (
            f"State atom count ({n_state}) != reward atom count ({n_reward}). "
            f"wrapper={wrapper_type.value}, structure={structure_fixture}"
        )
        assert torch.isfinite(reward_inputs.input_coords).all(), (
            f"Non-finite reward coordinates for "
            f"wrapper={wrapper_type.value}, structure={structure_fixture}"
        )
        assert torch.all(reward_inputs.occupancies > 0), (
            f"Non-positive reward occupancies for "
            f"wrapper={wrapper_type.value}, structure={structure_fixture}"
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", get_slow_wrappers(), ids=lambda w: w.value)
class TestRealWrapperNumericalStability:
    """Numerical stability tests with all real wrappers."""

    def test_trajectory_finite_throughout(
        self,
        wrapper_type: StructurePredictor,
        temp_output_dir,
        request,
    ):
        """All trajectory states are finite with real wrapper."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue("structure_1vme")
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(wrapper_type, structure, temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        sampler = create_sampler_from_type(TrajectorySamplers.AF3EDM, device=device)
        num_steps = 5
        schedule = sampler.compute_schedule(num_steps=num_steps)
        trajectory = []

        for i in range(num_steps):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(state, wrapper, context, features=features)
            state = step_output.state
            trajectory.append(state.clone())

        for i, traj_state in enumerate(trajectory):
            assert torch.isfinite(traj_state).all(), f"State at step {i} has NaN/Inf"
