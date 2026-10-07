"""Regression tests for full-sequence positions in Protenix inputs.

TODO: generalize these across all ModelWrappers"""

import numpy as np
import pytest
from atomworks.enums import ChainType
from biotite.structure import AtomArray, BondList, BondType


pytest.importorskip("protenix", reason="Protenix not installed")

from sampleworks.models.protenix.structure_processing import (
    get_poly_res_names,
    structure_to_protenix_json,
)


def test_valid_positions_preserve_insertion_coded_residues():
    """Retain every sequence position sharing a valid author residue number."""
    atoms = AtomArray(4)
    atoms.chain_id = np.array(["A"] * 4)
    atoms.res_id = np.array([42, 42, 42, 50])
    atoms.ins_code = np.array(["", "A", "B", ""])
    atoms.res_name = np.array(["MSE", "SEP", "TPO", "ALA"])
    atoms.atom_name = np.array(["CA"] * 4)
    atoms.set_annotation("label_entity_id", np.array(["1"] * 4))
    atoms.set_annotation("seq_idx", np.array([3, 4, 5, 9]))
    chain_info = {"A": {"chain_type": ChainType.POLYPEPTIDE_L}}

    result = get_poly_res_names(atoms, chain_info, valid_positions={"A": {42, 999}})

    assert result == {"1": [(4, "MSE"), (5, "SEP"), (6, "TPO")]}


def test_json_modifications_use_full_sequence_positions_with_insertions():
    """Serialize insertion-coded modifications without trimming missing termini."""
    atoms = AtomArray(3)
    atoms.coord = np.zeros((3, 3))
    atoms.chain_id = np.array(["A"] * 3)
    atoms.res_id = np.array([42, 42, 50])
    atoms.ins_code = np.array(["", "A", ""])
    atoms.res_name = np.array(["MSE", "SEP", "ALA"])
    atoms.atom_name = np.array(["CA"] * 3)
    atoms.element = np.array(["C"] * 3)
    atoms.set_annotation("seq_idx", np.array([3, 4, 9]))
    atoms.bonds = BondList(3)
    structure = {
        "asym_unit": atoms,
        "chain_info": {
            "A": {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "rcsb_entity": "1",
                "processed_entity_canonical_sequence": "GGGMSGGGGAGG",
            }
        },
    }

    result = structure_to_protenix_json(structure)

    protein = result["sequences"][0]["proteinChain"]
    assert protein["sequence"] == "GGGMSGGGGAGG"
    assert protein["modifications"] == [
        {"ptmPosition": 4, "ptmType": "CCD_MSE"},
        {"ptmPosition": 5, "ptmType": "CCD_SEP"},
    ]


@pytest.mark.parametrize(
    ("seq_idx", "expected_positions"),
    [([3, 8], (4, 9)), (None, (1, 9)), ([-1, -1], (1, 9))],
    ids=["full-sequence", "no-override", "unmapped-chain"],
)
def test_polymer_bond_endpoints_use_sequence_positions(seq_idx, expected_positions):
    """Use full-sequence indices for both polymer endpoints, retaining legacy fallback."""
    atoms = AtomArray(2)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([42, 50])
    atoms.res_name = np.array(["MSE", "CYS"])
    atoms.atom_name = np.array(["SE", "SG"])
    atoms.element = np.array(["Se", "S"])
    if seq_idx is not None:
        atoms.set_annotation("seq_idx", np.array(seq_idx))
    atoms.bonds = BondList(2, np.array([[0, 1, BondType.SINGLE]]))
    structure = {
        "asym_unit": atoms,
        "chain_info": {
            "A": {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "rcsb_entity": "1",
                "processed_entity_canonical_sequence": "GGGMGGGGCG",
            }
        },
    }

    result = structure_to_protenix_json(structure)

    assert result["covalent_bonds"] == [
        {
            "entity1": 1,
            "position1": expected_positions[0],
            "atom1": "SE",
            "entity2": 1,
            "position2": expected_positions[1],
            "atom2": "SG",
        }
    ]


@pytest.mark.parametrize("polymer_first", [True, False])
def test_ligand_bond_uses_sequence_position_only_for_polymer(polymer_first):
    """Apply seq_idx to either polymer endpoint while ligand positions stay one."""
    atoms = AtomArray(3)
    atoms.coord = np.zeros((3, 3))
    atoms.chain_id = np.array(["A", "B", "B"])
    atoms.res_id = np.array([42, 100, 100])
    atoms.res_name = np.array(["CYS", "LIG", "LIG"])
    atoms.atom_name = np.array(["SG", "C1", "C2"])
    atoms.element = np.array(["S", "C", "C"])
    atoms.set_annotation("seq_idx", np.array([4, -1, -1]))
    atoms.bonds = BondList(3, np.array([[0, 1, BondType.SINGLE]]))
    if not polymer_first:
        atoms = atoms[np.array([1, 2, 0])]
    structure = {
        "asym_unit": atoms,
        "chain_info": {
            "A": {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "rcsb_entity": "1",
                "processed_entity_canonical_sequence": "GGGGCGG",
            },
            "B": {"chain_type": ChainType.NON_POLYMER, "rcsb_entity": "2"},
        },
    }

    result = structure_to_protenix_json(structure)

    assert len(result["covalent_bonds"]) == 1
    bond = result["covalent_bonds"][0]
    endpoints = {(bond[f"entity{i}"], bond[f"position{i}"], bond[f"atom{i}"]) for i in (1, 2)}
    assert endpoints == {(1, 5, "SG"), (2, 1, "C1")}
