"""Tests for Boltz model wrappers.

Tests specific to Boltz1Wrapper and Boltz2Wrapper implementations,
including helper functions and model-specific behavior.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch


pytest.importorskip("boltz", reason="Boltz not installed")

from sampleworks.models.boltz.wrapper import (
    _atom_array_from_boltz_npz,
    _load_model_atom_array_from_structures_dir,
    BoltzConditioning,
    BoltzConfig,
    create_boltz_input_from_structure,
    process_structure_for_boltz,
)
from sampleworks.utils.guidance_constants import StructurePredictor
from tests.conftest import get_fixture_name_for_wrapper_type, STRUCTURES


BOLTZ_WRAPPER_TYPES = [StructurePredictor.BOLTZ_1, StructurePredictor.BOLTZ_2]


class TestCreateBoltzInputFromStructure:
    """Test the helper function that creates Boltz YAML from Atomworks structures."""

    def test_creates_yaml_file(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"
        assert (
            yaml_path.name == f"{structure_6b8x.get('metadata', {}).get('id', 'boltz_input')}.yaml"
        )

    def test_yaml_contains_sequences_key(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_protein_chain_in_yaml(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        content = yaml_path.read_text()

        for chain_id, chain_data in structure_6b8x["chain_info"].items():
            if chain_data["chain_type"].is_protein():
                assert f"id: {chain_id}" in content
                assert chain_data["processed_entity_canonical_sequence"] in content

    def test_sequence_override_in_yaml(self, structure_6b8x: dict, temp_output_dir: Path):
        """The sequence override should reach the YAML."""
        from sampleworks.utils.sequence import apply_sequence_override

        sequence = "ACDEFG"
        overridden = apply_sequence_override(structure_6b8x, sequence)
        yaml_path = create_boltz_input_from_structure(
            overridden,
            temp_output_dir,
            msa_manager=None,
            msa_pairing_strategy="greedy",
        )

        content = yaml_path.read_text()
        assert f"sequence: {sequence}" in content

    def test_handles_cif_format(self, structure_1vme: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_1vme, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_handles_pdb_format(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_works_with_test_structures(
        self, structure_fixture: str, temp_output_dir: Path, request
    ):
        structure = request.getfixturevalue(structure_fixture)
        yaml_path = create_boltz_input_from_structure(
            structure, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert len(content) > 0
        assert "sequences:" in content


class TestAnnotateStructureForBoltz:
    """Test the process_structure_for_boltz helper function."""

    def test_annotate_adds_boltz_config(self, structure_6b8x: dict, temp_output_dir: Path):
        result = process_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        assert "_boltz_config" in result
        assert isinstance(result["_boltz_config"], BoltzConfig)

    def test_annotate_preserves_original_structure(
        self, structure_6b8x: dict, temp_output_dir: Path
    ):
        result = process_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        for key in structure_6b8x:
            assert key in result
            assert result[key] is structure_6b8x[key]

    def test_annotate_default_values(self, structure_6b8x: dict, temp_output_dir: Path):
        result = process_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        config = result["_boltz_config"]
        assert config.num_workers == 0
        assert config.recycling_steps == 3

    def test_annotate_custom_values(self, structure_6b8x: dict, temp_output_dir: Path):
        result = process_structure_for_boltz(
            structure_6b8x,
            out_dir=temp_output_dir,
            num_workers=4,
            recycling_steps=5,
        )
        config = result["_boltz_config"]
        assert config.num_workers == 4
        assert config.recycling_steps == 5

    def test_annotate_out_dir_from_metadata(self, structure_6b8x: dict):
        result = process_structure_for_boltz(structure_6b8x)
        config = result["_boltz_config"]
        expected_id = structure_6b8x.get("metadata", {}).get("id", "boltz_output")
        assert config.out_dir == expected_id


class TestBoltzConfig:
    """Test the BoltzConfig dataclass."""

    def test_boltz_config_default_values(self):
        config = BoltzConfig()
        assert config.out_dir is None
        assert config.num_workers == 0
        assert config.recycling_steps == 3

    def test_boltz_config_custom_values(self, temp_output_dir: Path):
        config = BoltzConfig(
            out_dir=temp_output_dir,
            num_workers=4,
            recycling_steps=2,
        )
        assert config.out_dir == temp_output_dir
        assert config.num_workers == 4
        assert config.recycling_steps == 2


class TestBoltzNpzAtomArrayHelpers:
    """Unit tests for Boltz NPZ parsing and loading helpers."""

    def _write_npz(
        self,
        out_path: Path,
        *,
        include_chains: bool = True,
    ) -> Path:
        atoms_dtype = np.dtype([("coords", np.float32, (3,)), ("name", "U4")])
        residues_dtype = np.dtype([("atom_idx", np.int32), ("res_idx", np.int32), ("name", "U4")])
        chains_dtype = np.dtype([("res_idx", np.int32), ("name", "U2")])

        atoms = np.zeros(2, dtype=atoms_dtype)
        atoms["coords"] = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        atoms["name"] = np.array(["N", "CA"])

        residues = np.zeros(1, dtype=residues_dtype)
        residues["atom_idx"] = np.array([0], dtype=np.int32)
        residues["res_idx"] = np.array([7], dtype=np.int32)
        residues["name"] = np.array(["ALA"])

        chains = np.zeros(1, dtype=chains_dtype)
        chains["res_idx"] = np.array([0], dtype=np.int32)
        chains["name"] = np.array(["A"])

        if include_chains:
            np.savez(out_path, atoms=atoms, residues=residues, chains=chains)
        else:
            np.savez(out_path, atoms=atoms, residues=residues)
        return out_path

    def test_atom_array_from_npz(self, temp_output_dir: Path):
        npz_path = self._write_npz(temp_output_dir / "single.npz")

        arr = _atom_array_from_boltz_npz(npz_path)
        assert len(arr) == 2
        assert cast(np.ndarray, arr.chain_id).tolist() == ["A", "A"]
        assert cast(np.ndarray, arr.res_id).tolist() == [7, 7]
        assert cast(np.ndarray, arr.atom_name).tolist() == ["N", "CA"]
        assert arr.occupancy is not None
        assert arr.b_factor is not None

    def test_atom_array_from_npz_validates_missing_keys(self, temp_output_dir: Path):
        npz_path = self._write_npz(temp_output_dir / "missing_chains.npz", include_chains=False)
        with pytest.raises(ValueError, match="missing required keys"):
            _atom_array_from_boltz_npz(npz_path)

    def test_load_model_atom_array_from_structures_dir(self, temp_output_dir: Path):
        structures_dir = temp_output_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)
        self._write_npz(structures_dir / "target.npz")

        arr = _load_model_atom_array_from_structures_dir(structures_dir)
        assert arr is not None
        assert len(arr) == 2


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
class TestBoltzWrapperInitialization:
    """Test Boltz wrapper initialization and setup."""

    def test_wrapper_initializes(self, wrapper_type: StructurePredictor, request):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        assert wrapper is not None
        assert hasattr(wrapper, "model")
        assert hasattr(wrapper, "device")

    def test_model_on_correct_device(self, wrapper_type: StructurePredictor, device, request):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        assert wrapper.device == device
        assert next(wrapper.model.parameters()).device == device


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestBoltzWrapperFeaturize:
    """Test Boltz wrapper featurize method with all structures."""

    def test_featurize_returns_generative_model_input(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        assert isinstance(features, GenerativeModelInput)
        assert isinstance(features.conditioning, BoltzConditioning)
        assert not hasattr(features, "x_init")

    def test_featurize_creates_data_module(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        wrapper.featurize(annotated)
        assert hasattr(wrapper, "data_module")
        assert wrapper.data_module is not None


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestBoltzWrapperStep:
    """Test Boltz wrapper step method with all structures."""

    def test_step_returns_tensor(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([1.0])
        result = wrapper.step(state, t, features=features)

        assert torch.is_tensor(result)

    def test_step_output_shape_matches_input(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([1.0])
        result = wrapper.step(state, t, features=features)

        assert result.shape == state.shape
        assert result.shape[-1] == 3

    def test_step_with_high_t(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([160.0])
        result = wrapper.step(state, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == state.shape

    def test_step_with_low_t(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([0.01])
        result = wrapper.step(state, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == state.shape

    def test_step_with_float_t(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        result = wrapper.step(state, 1.0, features=features)

        assert torch.is_tensor(result)

    def test_step_requires_features(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="features"):
            wrapper.step(state, t, features=None)

    def test_step_denoises_input(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        t = torch.tensor([1.0])
        result = wrapper.step(state, t, features=features)

        assert not torch.allclose(result, state)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestBoltzWrapperInitializeFromPrior:
    """Test Boltz wrapper initialize_from_prior method with all structures."""

    # TODO: apply checking of this to all model wrappers once I figure out all the shape issues in
    # a more general way

    def test_initialize_from_prior_returns_tensor(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)

        result = wrapper.initialize_from_prior(batch_size=2, features=features)

        assert torch.is_tensor(result)

    def test_initialize_from_prior_correct_shape_from_features(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)

        batch_size = 3
        result = wrapper.initialize_from_prior(batch_size=batch_size, features=features)

        assert result.shape[0] == batch_size
        assert result.shape[-1] == 3

    def test_initialize_from_prior_batch_dimension(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)

        for batch_size in [1, 2, 5]:
            result = wrapper.initialize_from_prior(batch_size=batch_size, features=features)
            assert result.shape[0] == batch_size


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
class TestBoltzWrapperInitializeFromPriorValidation:
    """Test Boltz wrapper initialize_from_prior validation (no structure needed)."""

    def test_initialize_from_prior_correct_shape_from_explicit(
        self, wrapper_type: StructurePredictor, request
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        batch_size = 2
        num_atoms = 100
        result = wrapper.initialize_from_prior(batch_size=batch_size, shape=(num_atoms, 3))

        assert result.shape == (batch_size, num_atoms, 3)

    def test_initialize_from_prior_shape_validation(
        self, wrapper_type: StructurePredictor, request
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        with pytest.raises(ValueError, match="shape"):
            wrapper.initialize_from_prior(batch_size=2, shape=(100,))

        with pytest.raises(ValueError, match="shape"):
            wrapper.initialize_from_prior(batch_size=2, shape=(100, 4))

    def test_initialize_from_prior_requires_features_or_shape(
        self, wrapper_type: StructurePredictor, request
    ):
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        with pytest.raises(ValueError, match="features|shape"):
            wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", BOLTZ_WRAPPER_TYPES, ids=lambda w: w.value)
@pytest.mark.parametrize("structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", ""))
class TestBoltzWrappersEndToEnd:
    """End-to-end integration tests for Boltz wrappers with all structures."""

    def test_full_pipeline(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)

        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        assert isinstance(features, GenerativeModelInput)
        assert isinstance(features.conditioning, BoltzConditioning)

        state = wrapper.initialize_from_prior(batch_size=2, features=features)
        assert torch.is_tensor(state)
        assert state.shape[0] == 2

        t = torch.tensor([1.0])
        output = wrapper.step(state, t, features=features)
        assert torch.is_tensor(output)
        assert output.shape == state.shape
        assert torch.isfinite(output).all()

    def test_multiple_step_calls(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        """Test that multiple step() calls work correctly (simulating sampling loop)."""
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)

        annotated = process_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)

        x_t = wrapper.initialize_from_prior(batch_size=1, features=features)

        for t_val in [160.0, 80.0, 40.0, 20.0, 10.0]:
            t = torch.tensor([t_val])
            x_t = wrapper.step(x_t, t, features=features)
            assert torch.is_tensor(x_t)
            assert x_t.shape[-1] == 3
