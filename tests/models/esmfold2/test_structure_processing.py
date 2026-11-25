"""Atomworks structures translate faithfully into ESMFold2 inputs."""

from typing import Any

import pytest


pytest.importorskip(modname="esm.models.esmfold2.types", reason="ESMFold2 not installed")

import numpy as np
import torch
from biotite.structure import array, Atom, get_residue_starts
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from sampleworks.core.losses.protocol import LossInputs
from sampleworks.models.esmfold2.structure_processing import (
    esmfold2_features_to_atom_array,
    ESMFold2InputBuilderBlockLogs,
    structure_to_esmfold2_input,
)
from sampleworks.utils.sequence import expected_heavy_atom_count


# Loads the various structure dictionary fixtures into the namespace.
pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


@pytest.mark.parametrize("use_stack", [False, True], ids=["atom-array", "atom-array-stack"])
def test_single_chain_preserves_sequence_and_identity(
    fresh_structure_6b8x: dict[str, Any], use_stack: bool
):
    """Translate either atom-array form without modifying the source structure."""
    structure = fresh_structure_6b8x
    if not use_stack:
        structure["asym_unit"] = structure["asym_unit"][0]

    sequence = structure["chain_info"]["A"]["processed_entity_canonical_sequence"]

    result = structure_to_esmfold2_input(structure)

    assert isinstance(result, StructurePredictionInput)
    assert result.sequences == [ProteinInput(id="A", sequence=sequence)]
    assert result.pocket is None
    assert result.distogram_conditioning is None
    assert result.covalent_bonds is None


def test_prepared_input_builds_valid_features(
    structure_6b8x: dict[str, Any],
):
    """Build CPU features using ESMFold2's external CCD cache, without model weights."""
    model_input = structure_to_esmfold2_input(structure_6b8x)

    builder = ESMFold2InputBuilderBlockLogs()

    with pytest.warns(UserWarning, match="No MSA provided"):
        features, chains = builder.prepare_input(model_input, seed=0, device="cpu")

    atoms = structure_6b8x["asym_unit"][0]
    residue_names = atoms.res_name[get_residue_starts(atoms)].tolist()
    n_residues = len(residue_names)
    assert len(chains) == 1
    assert chains[0].chain_id == "A"
    assert [token.residue_name for token in chains[0].tokens] == residue_names
    assert features["token_attention_mask"].sum().item() == n_residues
    assert features["msa"].shape == (1, 1, n_residues)
    assert not features["has_deletion"].any()
    assert features["ref_pos"].shape[:2] == features["atom_attention_mask"].shape
    assert features["ref_pos"].shape[-1] == 3
    assert features["ref_pos"].device.type == "cpu"
    assert torch.isfinite(features["ref_pos"]).all()


def test_model_atoms_include_missing_cb_atoms(fresh_structure_6b8x: dict[str, Any]):
    """Model atoms include one CB per non-glycine residue despite their absence in the input."""
    structure = fresh_structure_6b8x
    input_atoms = structure["asym_unit"][0]
    sequence = structure["chain_info"]["A"]["processed_entity_canonical_sequence"]
    expected_cb_residues = [
        position for position, letter in enumerate(sequence, start=1) if letter != "G"
    ]
    expected_atom_count = expected_heavy_atom_count(sequence)
    structure["asym_unit"] = input_atoms[input_atoms.atom_name != "CB"]

    builder = ESMFold2InputBuilderBlockLogs()
    with pytest.warns(UserWarning, match="No MSA provided"):
        features, chains = builder.prepare_input(structure_to_esmfold2_input(structure), seed=0)

    model_atoms = esmfold2_features_to_atom_array(features, chains)

    np.testing.assert_array_equal(
        model_atoms.res_id[model_atoms.atom_name == "CB"], expected_cb_residues
    )
    assert len(model_atoms) == expected_atom_count


def test_atom_conversion_preserves_model_order_and_coordinates():
    """Conversion returns exactly the nine AG atoms in model order with their coordinates."""
    model_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="AG")])
    builder = ESMFold2InputBuilderBlockLogs()
    with pytest.warns(UserWarning, match="No MSA provided"):
        features, chains = builder.prepare_input(model_input, seed=0)

    # Native ESMFold2 atom order for alanine followed by glycine.
    expected_atoms = [
        (1, "ALA", "N", "N"),
        (1, "ALA", "CA", "C"),
        (1, "ALA", "C", "C"),
        (1, "ALA", "O", "O"),
        (1, "ALA", "CB", "C"),
        (2, "GLY", "N", "N"),
        (2, "GLY", "CA", "C"),
        (2, "GLY", "C", "C"),
        (2, "GLY", "O", "O"),
    ]

    # Padded model coordinates have shape (1, atoms, 3); each row is distinct.
    coordinates = torch.arange(features["ref_pos"].numel(), dtype=torch.float32).reshape(
        features["ref_pos"].shape
    )
    features["ref_pos"] = coordinates

    model_atoms = esmfold2_features_to_atom_array(features, chains)
    expected = array(
        [
            Atom(
                coordinates[0, i].numpy(),
                chain_id="A",
                res_id=res_id,
                res_name=res_name,
                atom_name=atom_name,
                element=element,
                b_factor=20.0,
                occupancy=1.0,
            )
            for i, (res_id, res_name, atom_name, element) in enumerate(expected_atoms)
        ]
    )
    assert model_atoms == expected
    assert len(model_atoms) == features["atom_attention_mask"].sum().item() == 9


def test_model_atoms_support_loss_inputs():
    """Model atoms provide finite B-factors and occupancies for loss preparation."""
    model_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="AG")])
    builder = ESMFold2InputBuilderBlockLogs()
    features, chains = builder.prepare_input(model_input, seed=0)

    model_atoms = esmfold2_features_to_atom_array(features, chains)

    # Defaults should be 20 for b-factor and 1 for occupancy
    np.testing.assert_array_equal(model_atoms.b_factor, np.full(len(model_atoms), 20.0))
    np.testing.assert_array_equal(model_atoms.occupancy, np.ones(len(model_atoms)))

    # Ensure LossInputs can be initialized.
    LossInputs.from_atom_array(model_atoms, ensemble_size=1, device="cpu")


def test_malformed_asym_unit_raises_error(fresh_structure_6b8x: dict[str, Any]):
    fresh_structure_6b8x["asym_unit"] = np.arange(10)
    with pytest.raises(TypeError, match="asym_unit"):
        structure_to_esmfold2_input(fresh_structure_6b8x)

    del fresh_structure_6b8x["asym_unit"]
    with pytest.raises(KeyError, match="asym_unit"):
        structure_to_esmfold2_input(fresh_structure_6b8x)


def test_sequence_mismatch_raises_error(fresh_structure_6b8x: dict[str, Any]):
    """Reject a declared sequence that differs from the observed residues."""
    chain_info = fresh_structure_6b8x["chain_info"]["A"]
    sequence = chain_info["processed_entity_canonical_sequence"]
    assert sequence.startswith("M")
    chain_info["processed_entity_canonical_sequence"] = "A" + sequence[1:]

    with pytest.raises(ValueError, match="canonical sequence does not match the observed residues"):
        structure_to_esmfold2_input(fresh_structure_6b8x)


def test_multichain_raises_not_implemented(structure_5sop: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="Multichain"):
        structure_to_esmfold2_input(structure_5sop)


def test_nonprotein_chain_raises_not_implemented(ligand_chain_5sop: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="Non L-protein"):
        structure_to_esmfold2_input(ligand_chain_5sop)
