"""Atomworks structures translate faithfully into ESMFold2 inputs."""

import copy
from typing import Any

import pytest


pytest.importorskip(modname="esm.models.esmfold2.types", reason="ESMFold2 not installed")

import numpy as np
import torch
from biotite.structure import array, Atom, get_residue_starts
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from sampleworks.models.esmfold2.structure_processing import (
    esmfold2_features_to_atom_array,
    ESMFold2InputBuilderBlockLogs,
    structure_to_esmfold2_input,
)
from sampleworks.utils.atom_reconciler import AtomReconciler


# Loads the various structure dictionary fixtures into the namespace.
pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


@pytest.mark.parametrize("use_stack", [False, True], ids=["atom-array", "atom-array-stack"])
def test_single_chain_preserves_sequence_and_identity(
    single_chain_structure: dict[str, Any], use_stack: bool
):
    """Translate either atom-array form without modifying the source structure."""
    structure = single_chain_structure
    if not use_stack:
        structure["asym_unit"] = structure["asym_unit"][0]

    original = copy.deepcopy(structure)
    sequence = structure["chain_info"]["A"]["processed_entity_canonical_sequence"]

    result = structure_to_esmfold2_input(structure)

    assert isinstance(result, StructurePredictionInput)
    assert result.sequences == [ProteinInput(id="A", sequence=sequence)]
    assert result.pocket is None
    assert result.distogram_conditioning is None
    assert result.covalent_bonds is None
    assert structure["asym_unit"] == original["asym_unit"]
    assert structure["chain_info"] == original["chain_info"]


def test_prepared_input_builds_valid_features(
    single_chain_structure: dict[str, Any],
):
    """Build CPU features using ESMFold2's external CCD cache, without model weights."""
    model_input = structure_to_esmfold2_input(single_chain_structure)

    builder = ESMFold2InputBuilderBlockLogs()

    with pytest.warns(UserWarning, match="No MSA provided"):
        features, chains = builder.prepare_input(model_input, seed=0, device="cpu")

    atoms = single_chain_structure["asym_unit"][0]
    residue_names = atoms.res_name[get_residue_starts(atoms)].tolist()
    n_residues = len(residue_names)
    assert len(chains) == 1
    assert chains[0].chain_id == "A"
    assert [token.residue_name for token in chains[0].tokens] == residue_names
    assert features["token_attention_mask"].sum().item() == n_residues
    assert features["input_ids"].shape == (1, n_residues)
    assert features["msa"].shape == (1, 1, n_residues)
    assert not features["has_deletion"].any()
    assert features["atom_attention_mask"].sum().item() > n_residues
    assert features["ref_pos"].shape[:2] == features["atom_attention_mask"].shape
    assert features["ref_pos"].shape[-1] == 3
    assert features["ref_pos"].device.type == "cpu"
    assert torch.isfinite(features["ref_pos"]).all()


@pytest.mark.parametrize(
    "missing_atoms", [False, True], ids=["complete", "missing-sidechain-atoms"]
)
def test_model_atoms_reconcile_with_reference(
    single_chain_structure: dict[str, Any], missing_atoms: bool
):
    """Match atom identities despite different ordering, numbering, or coverage."""
    structure = single_chain_structure
    atoms = structure["asym_unit"][0]
    if missing_atoms:
        atoms = atoms[atoms.atom_name != "CB"]

    # Reverse atoms within each residue while retaining residue order.
    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    indices = np.concatenate(
        [np.arange(start, stop)[::-1] for start, stop in zip(starts[:-1], starts[1:], strict=True)]
    )
    atoms = atoms[indices]
    atoms.res_id += 100
    structure["asym_unit"] = atoms

    builder = ESMFold2InputBuilderBlockLogs()
    with pytest.warns(UserWarning, match="No MSA provided"):
        features, chains = builder.prepare_input(structure_to_esmfold2_input(structure), seed=0)

    # Give every atom a unique coordinate to detect changes to model atom order.
    features["ref_pos"] = torch.arange(features["ref_pos"].numel(), dtype=torch.float32).reshape(
        features["ref_pos"].shape
    )
    model_atoms = esmfold2_features_to_atom_array(features, chains)
    mask = features["atom_attention_mask"][0].bool()
    np.testing.assert_array_equal(model_atoms.coord, features["ref_pos"][0, mask].numpy())
    assert len(model_atoms) == mask.sum().item()

    reconciler = AtomReconciler.from_arrays(model_atoms, atoms)
    assert reconciler.has_mismatch
    assert reconciler.n_common == len(atoms)
    matched_model = model_atoms[reconciler.model_indices.numpy()]
    matched_reference = atoms[reconciler.struct_indices.numpy()]
    for annotation in ("chain_id", "res_name", "atom_name", "element"):
        np.testing.assert_array_equal(
            matched_model.get_annotation(annotation), matched_reference.get_annotation(annotation)
        )

    # Compare residue positions independently of the reference's author numbering.
    np.testing.assert_array_equal(
        matched_model.res_id,
        np.repeat(np.arange(1, len(starts)), np.diff(starts))[reconciler.struct_indices.numpy()],
    )

    if missing_atoms:
        assert np.any(model_atoms.atom_name == "CB")
        assert reconciler.n_model > reconciler.n_struct


def test_atoms_retain_coordinates_through_conversion_and_reconciliation():
    """Each atom retains its XYZ coordinates through conversion and reconciliation."""
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
    reference = array(
        [
            Atom(
                coordinates[0, i].numpy(),
                chain_id="A",
                res_id=res_id,
                res_name=res_name,
                atom_name=atom_name,
                element=element,
            )
            for i, (res_id, res_name, atom_name, element) in enumerate(expected_atoms)
        ]
    )
    assert model_atoms == reference

    # Retain a reordered subset with different author residue numbering.
    reference = reference[[3, 1, 0, 8, 6, 5]]
    reference.res_id += 100
    reconciler = AtomReconciler.from_arrays(model_atoms, reference)
    assert reconciler.n_common == len(reference)
    np.testing.assert_array_equal(
        model_atoms.coord[reconciler.model_indices.numpy()],
        reference.coord[reconciler.struct_indices.numpy()],
    )


def test_malformed_asym_unit_raises_error(single_chain_structure: dict[str, Any]):
    single_chain_structure["asym_unit"] = np.arange(10)
    with pytest.raises(TypeError, match="asym_unit"):
        structure_to_esmfold2_input(single_chain_structure)

    del single_chain_structure["asym_unit"]
    with pytest.raises(KeyError, match="asym_unit"):
        structure_to_esmfold2_input(single_chain_structure)


def test_sequence_mismatch_raises_error(single_chain_structure: dict[str, Any]):
    """Reject a declared sequence that differs from the observed residues."""
    chain_info = single_chain_structure["chain_info"]["A"]
    sequence = chain_info["processed_entity_canonical_sequence"]
    assert sequence.startswith("M")
    chain_info["processed_entity_canonical_sequence"] = "A" + sequence[1:]

    with pytest.raises(ValueError, match="canonical sequence does not match the observed residues"):
        structure_to_esmfold2_input(single_chain_structure)


def test_multichain_raises_not_implemented(multichain_structure: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="Multichain"):
        structure_to_esmfold2_input(multichain_structure)


def test_nonprotein_chain_raises_not_implemented(ligand_structure: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="Non L-protein"):
        structure_to_esmfold2_input(ligand_structure)


def test_modified_residue_raises_not_implemented(modified_residue_structure: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="MSE.*modified/nonstandard"):
        structure_to_esmfold2_input(modified_residue_structure)


def test_crosslinks_raise_not_implemented(crosslinked_structure: dict[str, Any]):
    with pytest.raises(NotImplementedError, match="crosslinks"):
        structure_to_esmfold2_input(crosslinked_structure)
