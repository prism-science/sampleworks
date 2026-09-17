"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement FlowModelWrapper protocol.
"""

import numpy as np
import pytest
import torch
from biotite.sequence import ProteinSequence
from sampleworks.utils.sequence import apply_sequence_override, expected_heavy_atom_count

from tests.conftest import (
    annotate_structure_for_wrapper,
    ComponentInfo,
    get_conditioning_type,
    get_fixture_name_for_wrapper,
    MODEL_WRAPPER_REGISTRY,
    STRUCTURES,
)


def get_slow_wrapper_infos() -> list[ComponentInfo]:
    """Get ComponentInfo for all wrappers that require checkpoints."""
    return [info for info in MODEL_WRAPPER_REGISTRY.values() if info.requires_checkpoint]


def test_generative_model_input_carries_only_conditioning():
    """GenerativeModelInput leaves state initialization to the model wrapper."""
    from sampleworks.models.protocol import GenerativeModelInput

    conditioning = {"num_atoms": 3}
    features = GenerativeModelInput(conditioning=conditioning)

    assert features.conditioning is conditioning
    assert not hasattr(features, "x_init")


@pytest.mark.parametrize("wrapper_info", get_slow_wrapper_infos(), ids=lambda w: w.name)
class TestFlowModelWrapperProtocol:
    """Test that wrappers implement FlowModelWrapper protocol correctly.

    The FlowModelWrapper protocol requires:
    - featurize(structure: dict) -> GenerativeModelInput[C]
    - step(x_t, t, *, features) -> FlowOrEnergyBasedModelOutputT
    - initialize_from_prior(batch_size, features, *, shape) -> FlowOrEnergyBasedModelOutputT
    """

    def test_isinstance_flow_model_wrapper(self, wrapper_info: ComponentInfo, request):
        """Test wrapper implements FlowModelWrapper protocol."""
        from sampleworks.models.protocol import FlowModelWrapper

        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)
        assert isinstance(wrapper, FlowModelWrapper), (
            f"{wrapper_info.name} does not implement FlowModelWrapper protocol"
        )

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_featurize_returns_generative_model_input(
        self, wrapper_info: ComponentInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test featurize returns GenerativeModelInput with conditioning."""
        from sampleworks.models.protocol import GenerativeModelInput

        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)
        structure = request.getfixturevalue(structure_fixture)
        conditioning_type = get_conditioning_type(wrapper_info)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        assert isinstance(features, GenerativeModelInput), (
            f"{wrapper_info.name}.featurize must return GenerativeModelInput, got {type(features)}"
        )
        assert features.conditioning is not None, (
            f"{wrapper_info.name}.featurize returned None for conditioning"
        )
        assert isinstance(features.conditioning, conditioning_type), (
            f"{wrapper_info.name}.featurize conditioning must be {conditioning_type.__name__}, "
            f"got {type(features.conditioning)}"
        )

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_step_returns_tensor(
        self,
        wrapper_info: ComponentInfo,
        structure_fixture: str,
        batch_size: int,
        temp_output_dir,
        request,
    ):
        """Test step(x_t, t, features) returns coordinates tensor."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        t = torch.ones(batch_size)
        state = wrapper.initialize_from_prior(batch_size=batch_size, features=features)
        result = wrapper.step(state, t, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.step must return Tensor, got {type(result)}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_info.name}.step result last dim should be 3, got {result.shape[-1]}"
        )
        assert result.shape == state.shape, (
            f"{wrapper_info.name}.step output shape {result.shape} != input shape {state.shape}"
        )
        assert result.shape[0] == batch_size, (
            f"{wrapper_info.name}.step output batch should be {batch_size}, got {result.shape[0]}"
        )
        assert torch.isfinite(result).all(), f"{wrapper_info.name}.step returned non-finite values"

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_step_with_float_t(
        self, wrapper_info: ComponentInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test step works with float t value."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        t = 1.0
        state = wrapper.initialize_from_prior(batch_size=1, features=features)
        result = wrapper.step(state, t, features=features)

        assert torch.is_tensor(result), f"{wrapper_info.name}.step must return Tensor with float t"

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_initialize_from_prior_with_features(
        self, wrapper_info: ComponentInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test initialize_from_prior with features from featurize."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        assert not hasattr(features, "x_init")
        batch_size = 3
        result = wrapper.initialize_from_prior(batch_size, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.initialize_from_prior must return Tensor"
        )
        assert result.shape[0] == batch_size, (
            f"{wrapper_info.name}.initialize_from_prior batch should be {batch_size}, "
            f"got {result.shape[0]}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_info.name}.initialize_from_prior last dim should be 3, got "
            f"{result.shape[-1]}"
        )

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_initialize_from_prior_with_shape(self, wrapper_info: ComponentInfo, request):
        """Test initialize_from_prior with explicit shape."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)

        batch_size = 2
        num_atoms = 100
        result = wrapper.initialize_from_prior(batch_size, shape=(num_atoms, 3))

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.initialize_from_prior must return Tensor"
        )
        assert result.shape == (batch_size, num_atoms, 3), (
            f"{wrapper_info.name}.initialize_from_prior shape should be "
            f"({batch_size}, {num_atoms}, 3), got {result.shape}"
        )

    def test_initialize_from_prior_raises_without_features_or_shape(
        self, wrapper_info: ComponentInfo, request
    ):
        """Test initialize_from_prior raises ValueError without features or shape."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)

        with pytest.raises(ValueError, match="features|shape"):
            wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.parametrize("wrapper_info", get_slow_wrapper_infos(), ids=lambda w: w.name)
class TestStepRequiresFeatures:
    """Test that step() requires features parameter."""

    def test_step_raises_without_features(self, wrapper_info: ComponentInfo, request):
        """Test step raises ValueError when features is None."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)

        x_t = torch.randn(1, 100, 3, device=wrapper.device)
        t = torch.tensor([1.0])

        with pytest.raises(ValueError, match="features"):
            wrapper.step(x_t, t, features=None)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_info", get_slow_wrapper_infos(), ids=lambda w: w.name)
class TestFullSequenceAtomCoverage:
    """Verify that preprocessing with the real deposited sequence override produces
    features containing atoms for ALL residues and not just the observed residues.

    Uses the 5I09 density-input fixture (single protein chain, 366 observed out of
    386 deposited residues: 9 N-terminal, 8 internal, and 3 C-terminal gaps).
    The override uses the actual PDB deposited sequence so that the alignment
    must handle real gap positions.
    """

    def test_featurize_includes_all_sequence_residues(
        self,
        wrapper_info: ComponentInfo,
        structure_5i09_density: dict,
        seq_5i09_deposited: str,
        temp_output_dir,
        request,
    ):
        """After override with the real 386-residue deposited sequence,
        initialize_from_prior should produce more atoms than observed-only."""
        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)

        overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)

        # separate dirs so Boltz process_inputs doesn't cache collide
        full_seq_dir = temp_output_dir / "full_seq"
        full_seq_dir.mkdir(exist_ok=True)
        annotated = annotate_structure_for_wrapper(wrapper_info, overridden, full_seq_dir)
        features = wrapper.featurize(annotated)
        prior = wrapper.initialize_from_prior(batch_size=1, features=features)

        # The prior tensor's atom dimension should be consistent with the full
        # 386-residue deposited sequence, not the 366 observed residues.
        observed_dir = temp_output_dir / "observed_only"
        observed_dir.mkdir(exist_ok=True)
        observed_only = annotate_structure_for_wrapper(
            wrapper_info, structure_5i09_density, observed_dir
        )
        features_observed = wrapper.featurize(observed_only)
        prior_observed = wrapper.initialize_from_prior(batch_size=1, features=features_observed)

        n_atoms_full = prior.shape[1]
        n_atoms_observed = prior_observed.shape[1]
        n_atoms_expected = expected_heavy_atom_count(seq_5i09_deposited)
        assert n_atoms_full > n_atoms_observed, (
            f"{wrapper_info.name}: full-sequence prior ({n_atoms_full} atoms) should have "
            f"more atoms than observed-only ({n_atoms_observed} atoms)"
        )
        # Some models add a C-terminal OXT (+1 atom); tolerate that.
        assert n_atoms_expected <= n_atoms_full <= n_atoms_expected + 1, (
            f"{wrapper_info.name}: full-sequence prior has {n_atoms_full} atoms, "
            f"expected {n_atoms_expected}(+1 OXT) heavy atoms from the deposited sequence"
        )

    def test_trajectory_serialization_preserves_full_sequence_atom_assignments(
        self,
        wrapper_info: ComponentInfo,
        structure_5i09_density: dict,
        seq_5i09_deposited: str,
        temp_output_dir,
        request,
    ):
        """The trajectory output template contains every full-sequence residue and atom.

        This follows the same featurization, trajectory-input processing, and trajectory
        serialization path used by ``run_guidance``. In particular, it verifies that
        sequence-override placeholders do not collide with observed residue IDs or merge
        atoms from different sequence positions into one output residue.
        """
        from biotite.structure.info import residue as ccd_residue
        from sampleworks.utils.atom_array_utils import parse_structure
        from sampleworks.utils.guidance_constants import GuidanceType
        from sampleworks.utils.guidance_script_utils import save_trajectory
        from sampleworks.utils.structure_utils import (
            get_asym_unit_from_structure,
            process_structure_to_trajectory_input,
        )

        fixture_name = get_fixture_name_for_wrapper(wrapper_info)
        wrapper = request.getfixturevalue(fixture_name)

        overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)
        annotated = annotate_structure_for_wrapper(wrapper_info, overridden, temp_output_dir)
        features = wrapper.featurize(annotated)
        prior = wrapper.initialize_from_prior(batch_size=1, features=features)

        processed = process_structure_to_trajectory_input(
            structure=annotated,
            coords_from_prior=prior,
            features=features,
            ensemble_size=1,
        )
        assert processed.model_atom_array is not None, (
            f"{wrapper_info.name}: full-sequence runs must expose a model atom template "
            "for trajectory serialization"
        )
        model_atom_array = processed.model_atom_array
        assert len(model_atom_array) == prior.shape[1], (
            f"{wrapper_info.name}: model atom template ({len(model_atom_array)}) and "
            f"trajectory ({prior.shape[1]}) have different atom counts"
        )

        save_trajectory(
            scaler_type=GuidanceType.PURE_GUIDANCE,
            trajectory=[prior.detach().cpu()],
            atom_array=model_atom_array,
            output_dir=temp_output_dir,
            subdir_name="denoised",
            save_every=1,
        )
        written = parse_structure(temp_output_dir / "trajectory" / "denoised" / "trajectory_0.cif")
        written_atom_array = get_asym_unit_from_structure(written, atom_array_index=0)

        assert len(written_atom_array) == len(model_atom_array)
        for annotation in ("chain_id", "res_id", "res_name", "atom_name", "element"):
            np.testing.assert_array_equal(
                getattr(written_atom_array, annotation),
                getattr(model_atom_array, annotation),
                err_msg=(
                    f"{wrapper_info.name}: {annotation} changed during trajectory serialization"
                ),
            )

        protein_chain = next(
            chain_id
            for chain_id, info in overridden["chain_info"].items()
            if info["chain_type"].is_protein()
        )
        expected_res_names = [
            ProteinSequence.convert_letter_1to3(letter) for letter in seq_5i09_deposited
        ]
        for array, label in ((model_atom_array, "model"), (written_atom_array, "written")):
            chain_mask = np.asarray(array.chain_id) == protein_chain
            assert np.all(array.occupancy[chain_mask] > 0), (
                f"{wrapper_info.name}: {label} output contains unoccupied protein atoms"
            )
            residue_names: list[str] = []
            residue_atom_names: dict[int, list[str]] = {}
            for res_id, res_name, atom_name in zip(
                np.asarray(array.res_id)[chain_mask],
                np.asarray(array.res_name)[chain_mask],
                np.asarray(array.atom_name)[chain_mask],
            ):
                res_id = int(res_id)
                if res_id not in residue_atom_names:
                    residue_names.append(str(res_name))
                    residue_atom_names[res_id] = []
                residue_atom_names[res_id].append(str(atom_name))

            assert residue_names == expected_res_names, (
                f"{wrapper_info.name}: {label} output does not contain the full sequence "
                "with one residue assignment per sequence position"
            )
            for position, (res_id, res_name) in enumerate(zip(residue_atom_names, residue_names)):
                expected_atom_names = {
                    str(atom_name)
                    for element, atom_name in zip(
                        ccd_residue(res_name).element, ccd_residue(res_name).atom_name
                    )
                    if str(element) != "H" and str(atom_name) != "OXT"
                }
                atom_names = residue_atom_names[res_id]
                actual_atom_names = set(atom_names)
                assert len(atom_names) == len(actual_atom_names), (
                    f"{wrapper_info.name}: {label} residue {position + 1} contains duplicate atoms"
                )
                allowed_atom_names = [expected_atom_names]
                if position == len(residue_names) - 1:
                    allowed_atom_names.append(expected_atom_names | {"OXT"})
                assert any(actual_atom_names == allowed for allowed in allowed_atom_names), (
                    f"{wrapper_info.name}: {label} residue {position + 1} has incorrect "
                    "atom assignments"
                )
