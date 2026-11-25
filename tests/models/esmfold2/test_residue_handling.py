from typing import Any

import numpy as np
import pytest
from sampleworks.models.esmfold2.residue_handling import canonicalize_residue_sequence


pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


def test_mse_structure_produces_canonical_sequence(protein_chain_1vme: dict[str, Any]):
    """MSE is canonicalized in the sequence while reference atoms remain unchanged."""
    atoms = protein_chain_1vme["asym_unit"][0]
    original = atoms.copy()
    assert "MSE" in atoms.res_name

    sequence = canonicalize_residue_sequence(atoms)

    chain_id = str(atoms.chain_id[0])
    expected_sequence = protein_chain_1vme["chain_info"][chain_id][
        "processed_entity_canonical_sequence"
    ]
    assert sequence == expected_sequence
    np.testing.assert_array_equal(atoms.coord, original.coord)
    for annotation in original.get_annotation_categories():
        np.testing.assert_array_equal(
            atoms.get_annotation(annotation), original.get_annotation(annotation)
        )
    assert atoms.bonds == original.bonds


@pytest.mark.parametrize("residue_name", ["CSO", "UNK"])
def test_other_nonstandard_residues_remain_unsupported(
    protein_chain_1vme: dict[str, Any], residue_name: str
):
    """Accepting MSE does not silently canonicalize other modified or unknown residues."""
    atoms = protein_chain_1vme["asym_unit"]
    atoms.res_name[atoms.res_name == "MSE"] = residue_name
    with pytest.raises(NotImplementedError, match=f"{residue_name}.*modified/nonstandard"):
        canonicalize_residue_sequence(atoms[0])
