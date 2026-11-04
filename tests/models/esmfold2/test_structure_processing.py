"""Atomworks structures translate faithfully into ESMFold2 inputs."""

import pytest


pytest.importorskip(modname="esm.models.esmfold2.types", reason="ESMFold2 not installed")

import numpy as np
from sampleworks.models.esmfold2.structure_processing import structure_to_esmfold2_input


pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


def test_malformed_asym_unit_raises_error(single_chain_structure):
    single_chain_structure["asym_unit"] = np.arange(10)
    with pytest.raises(TypeError, match="asym_unit"):
        structure_to_esmfold2_input(single_chain_structure)

    del single_chain_structure["asym_unit"]
    with pytest.raises(KeyError, match="asym_unit"):
        structure_to_esmfold2_input(single_chain_structure)


def test_sequence_mismatch_raises_error(single_chain_structure):
    """Reject a declared sequence that differs from the observed residues."""
    chain_info = single_chain_structure["chain_info"]["A"]
    sequence = chain_info["processed_entity_canonical_sequence"]
    assert sequence.startswith("M")
    chain_info["processed_entity_canonical_sequence"] = "A" + sequence[1:]

    with pytest.raises(ValueError, match="canonical sequence does not match the observed residues"):
        structure_to_esmfold2_input(single_chain_structure)


def test_multichain_raises_not_implemented(multichain_structure):
    with pytest.raises(NotImplementedError, match="Multichain"):
        structure_to_esmfold2_input(multichain_structure)


def test_nonprotein_chain_raises_not_implemented(ligand_structure):
    with pytest.raises(NotImplementedError, match="Non L-protein"):
        structure_to_esmfold2_input(ligand_structure)


def test_modified_residue_raises_not_implemented(modified_residue_structure):
    with pytest.raises(NotImplementedError, match="MSE.*modified/nonstandard"):
        structure_to_esmfold2_input(modified_residue_structure)


def test_crosslinks_raise_not_implemented(crosslinked_structure):
    with pytest.raises(NotImplementedError, match="crosslinks"):
        structure_to_esmfold2_input(crosslinked_structure)
