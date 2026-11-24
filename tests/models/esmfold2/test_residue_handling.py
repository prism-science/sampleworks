from typing import Any

import pytest
from sampleworks.models.esmfold2.residue_handling import canonicalize_residue_sequence


pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


def test_mse_structure_produces_canonical_sequence(modified_residue_structure: dict[str, Any]):
    """MSE is canonicalized in the sequence while reference atoms remain unchanged."""
    atoms = modified_residue_structure["asym_unit"][0]
    original = atoms.copy()
    assert "MSE" in atoms.res_name

    sequence = canonicalize_residue_sequence(atoms)

    chain_id = str(atoms.chain_id[0])
    expected_sequence = modified_residue_structure["chain_info"][chain_id][
        "processed_entity_canonical_sequence"
    ]
    assert sequence == expected_sequence
    assert atoms == original


@pytest.mark.parametrize("residue_name", ["CSO", "UNK"])
def test_other_nonstandard_residues_remain_unsupported(
    modified_residue_structure: dict[str, Any], residue_name: str
):
    """Accepting MSE does not silently canonicalize other modified or unknown residues."""
    atoms = modified_residue_structure["asym_unit"]
    atoms.res_name[atoms.res_name == "MSE"] = residue_name
    with pytest.raises(NotImplementedError, match=f"{residue_name}.*modified/nonstandard"):
        canonicalize_residue_sequence(atoms[0])
