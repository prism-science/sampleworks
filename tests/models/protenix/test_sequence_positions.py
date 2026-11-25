"""Regression tests for full-sequence positions in Protenix inputs.

TODO: generalize these across all ModelWrappers"""

from typing import Any

import numpy as np
import pytest
from atomworks.enums import ChainType
from biotite.structure import AtomArray, BondList, BondType


pytest.importorskip("protenix", reason="Protenix not installed")

from sampleworks.models.protenix.structure_processing import structure_to_protenix_json


LIGAND_INFO = {"chain_type": ChainType.NON_POLYMER}


def _atoms(chain_id, res_id, res_name, atom_name, bonds=(), **annotations) -> AtomArray:
    """Build one-model atoms at the origin with full occupancy.

    Parameters
    ----------
    chain_id, res_id, res_name, atom_name : Sequence
        Per-atom annotations, all the same length.
    bonds : Sequence[tuple[int, int, BondType]]
        Bonds as (atom_i, atom_j, bond_type).
    **annotations
        Extra per-atom annotations, e.g. ``ins_code`` or ``seq_idx``.

    Returns
    -------
    AtomArray
        The atoms, with a ``BondList``.
    """
    n_atoms = len(chain_id)
    atoms = AtomArray(n_atoms)
    atoms.coord = np.zeros((n_atoms, 3))
    atoms.chain_id = np.array(chain_id)
    atoms.res_id = np.array(res_id)
    atoms.res_name = np.array(res_name)
    atoms.atom_name = np.array(atom_name)
    atoms.set_annotation("occupancy", np.ones(n_atoms))
    for name, values in annotations.items():
        atoms.set_annotation(name, np.array(values))
    atoms.bonds = BondList(n_atoms, np.array(bonds, dtype=np.int64).reshape(-1, 3))
    return atoms


def _protein_info(sequence: str, res_id: list[int] | None = None) -> dict:
    """Build a protein ``chain_info`` entry.

    Parameters
    ----------
    sequence : str
        Deposited one-letter sequence.
    res_id : list[int] | None
        Residue numbering of the deposited sequence, if any.

    Returns
    -------
    dict
        The ``chain_info`` entry.
    """
    info: dict[str, Any] = {
        "chain_type": ChainType.POLYPEPTIDE_L,
        "processed_entity_canonical_sequence": sequence,
    }
    if res_id is not None:
        info["res_id"] = res_id
    return info


def test_modifications_use_full_sequence_positions_with_insertions():
    """Serialize insertion-coded modifications without trimming missing termini."""
    atoms = _atoms(
        ["A"] * 3,
        [42, 42, 50],
        ["MSE", "SEP", "ALA"],
        ["CA"] * 3,
        ins_code=["", "A", ""],
        seq_idx=[3, 4, 9],
    )
    structure = {"asym_unit": atoms, "chain_info": {"A": _protein_info("GGGMSGGGGAGG")}}

    result = structure_to_protenix_json(structure)

    protein = result["sequences"][0]["proteinChain"]
    assert protein["sequence"] == "GGGMSGGGGAGG"
    assert protein["modifications"] == [
        {"ptmPosition": 4, "ptmType": "CCD_MSE"},
        {"ptmPosition": 5, "ptmType": "CCD_SEP"},
    ]


def test_unresolved_modified_residue_is_not_a_modification():
    """List every resolved insertion-coded modification and skip an unresolved one."""
    atoms = _atoms(
        ["A"] * 4,
        [42, 42, 42, 50],
        ["MSE", "SEP", "TPO", "MLY"],
        ["CA"] * 4,
        ins_code=["", "A", "B", ""],
        seq_idx=[3, 4, 5, 9],
    )
    atoms.coord[3] = np.nan
    structure = {"asym_unit": atoms, "chain_info": {"A": _protein_info("GGGMSTGGGKG")}}

    result = structure_to_protenix_json(structure)

    protein = result["sequences"][0]["proteinChain"]
    assert protein["sequence"] == "GGGMSTGGGKG"
    assert protein["modifications"] == [
        {"ptmPosition": 4, "ptmType": "CCD_MSE"},
        {"ptmPosition": 5, "ptmType": "CCD_SEP"},
        {"ptmPosition": 6, "ptmType": "CCD_TPO"},
    ]


@pytest.mark.parametrize(
    "seq_idx", [[3, 8], None, [-1, -1]], ids=["override", "chain-info-numbering", "unplaced"]
)
def test_polymer_bond_endpoints_use_sequence_positions(seq_idx):
    """Place both polymer endpoints by deposited-sequence position, not author numbering."""
    annotations = {} if seq_idx is None else {"seq_idx": seq_idx}
    atoms = _atoms(
        ["A", "A"], [42, 50], ["MSE", "CYS"], ["SE", "SG"], [(0, 1, BondType.SINGLE)], **annotations
    )
    numbering = [39, 40, 41, 42, 43, 44, 45, 46, 50, 51]
    structure = {"asym_unit": atoms, "chain_info": {"A": _protein_info("GGGMGGGGCG", numbering)}}

    result = structure_to_protenix_json(structure)

    assert result["covalent_bonds"] == [
        {
            "entity1": 1,
            "position1": 4,
            "atom1": "SE",
            "entity2": 1,
            "position2": 9,
            "atom2": "SG",
        }
    ]


@pytest.mark.parametrize(("third_residue", "expect_bond"), [("MSE", True), ("MET", False)])
def test_polymer_polymer_bond_needs_a_modification(third_residue, expect_bond):
    """Emit a disulfide only when some polymer residue is modified."""
    atoms = _atoms(
        ["A"] * 3,
        [1, 5, 6],
        ["CYS", "CYS", third_residue],
        ["SG", "SG", "CA"],
        [(0, 1, BondType.SINGLE)],
    )
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": _protein_info("CGGGCM", [1, 2, 3, 4, 5, 6])},
    }

    result = structure_to_protenix_json(structure)

    expected_bonds = [
        {"entity1": 1, "position1": 1, "atom1": "SG", "entity2": 1, "position2": 5, "atom2": "SG"}
    ]
    assert result.get("covalent_bonds") == (expected_bonds if expect_bond else None)


@pytest.mark.parametrize("polymer_first", [True, False])
@pytest.mark.parametrize(
    ("bond_type", "expected_order"),
    [(BondType.SINGLE, None), (BondType.DOUBLE, 2), (BondType.TRIPLE, 3)],
)
def test_ligand_bond_positions_and_order(polymer_first, bond_type, expected_order):
    """Use seq_idx for the polymer endpoint, one for the ligand, and keep high bond orders."""
    atoms = _atoms(
        ["A", "B", "B"],
        [42, 100, 100],
        ["CYS", "LIG", "LIG"],
        ["SG", "C1", "C2"],
        [(0, 1, bond_type)],
        seq_idx=[4, -1, -1],
    )
    if not polymer_first:
        atoms = atoms[np.array([1, 2, 0])]
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": _protein_info("GGGGCGG"), "B": LIGAND_INFO},
    }

    result = structure_to_protenix_json(structure)

    protein_entity = 1 if polymer_first else 2
    ligand_entity = 3 - protein_entity
    assert "proteinChain" in result["sequences"][protein_entity - 1]
    assert len(result["covalent_bonds"]) == 1
    bond = result["covalent_bonds"][0]
    endpoints = {(bond[f"entity{i}"], bond[f"position{i}"], bond[f"atom{i}"]) for i in (1, 2)}
    assert endpoints == {(protein_entity, 5, "SG"), (ligand_entity, 1, "C1")}
    assert bond.get("bondOrder") == expected_order


@pytest.mark.parametrize("unresolved", ["absent", "nan-coords"])
def test_unresolved_n_terminus_keeps_full_sequence(unresolved):
    """Model unresolved leading residues and shift modification and bond positions to match."""
    atoms = _atoms(
        ["A", "A", "A", "A", "A", "B", "B"],
        [1, 2, 3, 4, 5, 10, 10],
        ["MET", "LYS", "SEP", "CYS", "GLY", "LIG", "LIG"],
        ["CA", "CA", "P", "SG", "CA", "C1", "C2"],
        [(0, 1, BondType.SINGLE), (3, 5, BondType.SINGLE)],
    )
    if unresolved == "absent":
        atoms = atoms[2:]
    else:
        atoms.coord[:2] = np.nan
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": _protein_info("MKSCG", [1, 2, 3, 4, 5]), "B": LIGAND_INFO},
    }

    result = structure_to_protenix_json(structure)

    assert result["sequences"][0]["proteinChain"] == {
        "sequence": "MKSCG",
        "msa": {},
        "modifications": [{"ptmPosition": 3, "ptmType": "CCD_SEP"}],
        "count": 1,
    }
    assert result["covalent_bonds"] == [
        {"entity1": 1, "position1": 4, "atom1": "SG", "entity2": 2, "position2": 1, "atom2": "C1"}
    ]


def test_homodimer_shares_one_entity_and_merges_bonds():
    """Group identical chains under one counted entity and merge a bond present in each copy."""
    atoms = _atoms(
        ["A"] * 3 + ["B"] * 3 + ["C", "C", "D", "D"],
        [1, 2, 3, 1, 2, 3, 1, 1, 1, 1],
        ["GLY", "CYS", "GLY"] * 2 + ["LIG"] * 4,
        ["CA", "SG", "CA"] * 2 + ["C1", "C2"] * 2,
        [(1, 6, BondType.DOUBLE), (4, 8, BondType.DOUBLE)],
    )
    protein = _protein_info("GCG")
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": protein, "B": protein, "C": LIGAND_INFO, "D": LIGAND_INFO},
    }

    result = structure_to_protenix_json(structure)

    assert result["sequences"] == [
        {"proteinChain": {"sequence": "GCG", "msa": {}, "count": 2}},
        {"ligand": {"ligand": "CCD_LIG", "count": 2}},
    ]
    assert result["covalent_bonds"] == [
        {
            "entity1": 1,
            "position1": 2,
            "atom1": "SG",
            "entity2": 2,
            "position2": 1,
            "atom2": "C1",
            "bondOrder": 2,
        }
    ]


def test_multi_residue_ligand_positions_are_residue_ordinals():
    """Position each ligand residue by its order in the ligand chain."""
    atoms = _atoms(
        ["A", "B", "B", "B"],
        [1, 201, 201, 202],
        ["ASN", "NAG", "NAG", "NAG"],
        ["ND2", "C1", "O4", "C1"],
        [(0, 1, BondType.SINGLE), (2, 3, BondType.SINGLE)],
    )
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": _protein_info("N"), "B": LIGAND_INFO},
    }

    result = structure_to_protenix_json(structure)

    assert result["sequences"][1] == {"ligand": {"ligand": "CCD_NAG_NAG", "count": 1}}
    assert result["covalent_bonds"] == [
        {"entity1": 1, "position1": 1, "atom1": "ND2", "entity2": 2, "position2": 1, "atom2": "C1"},
        {"entity1": 2, "position1": 1, "atom1": "O4", "entity2": 2, "position2": 2, "atom2": "C1"},
    ]
