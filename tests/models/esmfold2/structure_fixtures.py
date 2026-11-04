"""Atomworks structure fixtures for ESMFold2 tests."""

from pathlib import Path
from typing import Any

import pytest
from atomworks import parse


@pytest.fixture
def single_chain_structure(resources_dir: Path) -> dict[str, Any]:
    """Parse the single-chain 6B8X structure with production parser options."""
    return parse(
        filename=resources_dir / "6b8x" / "6b8x_final.pdb",
        hydrogen_policy="remove",
        add_missing_atoms=False,
    )


@pytest.fixture
def multichain_structure(resources_dir: Path) -> dict[str, Any]:
    """Parse both 5SOP protein chains, excluding the RYI ligand."""
    return parse(
        filename=resources_dir / "5SOP" / "5SOP_single_001.pdb",
        hydrogen_policy="remove",
        add_missing_atoms=False,
        remove_ccds=["RYI"],
    )


@pytest.fixture
def protein_ligand_structure(resources_dir: Path) -> dict[str, Any]:
    """Parse 5SOP with its protein chains and RYI ligand."""
    return parse(
        filename=resources_dir / "5SOP" / "5SOP_single_001.pdb",
        hydrogen_policy="remove",
        add_missing_atoms=False,
    )


@pytest.fixture
def ligand_structure(protein_ligand_structure: dict[str, Any]) -> dict[str, Any]:
    """Select the RYI ligand chain from 5SOP.

    Parameters
    ----------
    protein_ligand_structure
        Parsed 5SOP structure containing the protein and ligand chains.

    Returns
    -------
    dict
        Structure containing only ligand chain C.
    """
    atoms = protein_ligand_structure["asym_unit"]
    return {
        **protein_ligand_structure,
        "asym_unit": atoms[:, atoms.chain_id == "C"],
        "chain_info": {"C": protein_ligand_structure["chain_info"]["C"]},
    }


@pytest.fixture
def modified_residue_structure(resources_dir: Path) -> dict[str, Any]:
    """Parse the carved 1VME protein chain containing selenomethionine (MSE)."""
    return parse(
        filename=resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif",
        hydrogen_policy="remove",
        add_missing_atoms=False,
    )


@pytest.fixture
def crosslinked_structure(resources_dir: Path) -> dict[str, Any]:
    """Parse crambin (1CRN) with its three disulfide bonds.

    Parameters
    ----------
    resources_dir
        Directory of checked-in test structures.

    Returns
    -------
    dict
        One protein chain with Cys3–Cys40, Cys4–Cys32, and Cys16–Cys26 bonds.

    Notes
    -----
    CIF source: https://files.rcsb.org/download/1CRN.cif.
    Atomworks excludes disulfide connections by default, so include them explicitly.
    """
    return parse(
        filename=resources_dir / "1CRN" / "1CRN.cif",
        hydrogen_policy="remove",
        add_missing_atoms=False,
        add_bond_types_from_struct_conn=["covale", "disulf"],
    )


@pytest.fixture(params=[False, True], ids=["observed-atoms", "with-unresolved-atoms"])
def missing_atoms_structure(resources_dir: Path, request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parse the damaged 5I09 prediction with and without atom completion."""
    return parse(
        filename=resources_dir / "5I09" / "5I09_prediction_deleted_atoms.cif",
        hydrogen_policy="remove",
        add_missing_atoms=request.param,
    )


@pytest.fixture(params=["remove", "keep"], ids=["without-hydrogens", "with-hydrogens"])
def hydrogen_structure(resources_dir: Path, request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parse the 5SOP protein chains with and without deposited hydrogens."""
    return parse(
        filename=resources_dir / "5SOP" / "5SOP_single_001.pdb",
        hydrogen_policy=request.param,
        add_missing_atoms=False,
        remove_ccds=["RYI"],
    )
