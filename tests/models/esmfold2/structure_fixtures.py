"""Independent copies and chain selections of shared structure fixtures."""

import copy
from typing import Any

import pytest


@pytest.fixture(scope="function")
def fresh_structure_6b8x(structure_6b8x: dict[str, Any]) -> dict[str, Any]:
    """Copy shared 6B8X data so tests can modify it independently."""
    return copy.deepcopy(structure_6b8x)


@pytest.fixture(scope="function")
def fresh_structure_5sop(structure_5sop: dict[str, Any]) -> dict[str, Any]:
    """Copy shared 5SOP data, containing protein chains A and B and RYI ligand chain C."""
    return copy.deepcopy(structure_5sop)


@pytest.fixture(scope="session")
def ligand_chain_5sop(structure_5sop: dict[str, Any]) -> dict[str, Any]:
    """Restrict the shared 5SOP asymmetric unit and chain metadata to RYI ligand chain C."""
    atoms = structure_5sop["asym_unit"]
    return {
        "asym_unit": atoms[:, atoms.chain_id == "C"],
        "chain_info": {"C": structure_5sop["chain_info"]["C"]},
        "assemblies": structure_5sop["assemblies"],
        "metadata": structure_5sop["metadata"],
        "extra_info": structure_5sop["extra_info"],
        "ligand_info": structure_5sop["ligand_info"],
    }


@pytest.fixture(scope="function")
def protein_chain_1vme(structure_1vme: dict[str, Any]) -> dict[str, Any]:
    """Copy shared 1VME data and select chain A, which contains seven MSE residues."""
    structure = copy.deepcopy(structure_1vme)
    atoms = structure["asym_unit"]
    structure["asym_unit"] = atoms[:, atoms.chain_id == "A"]
    structure["chain_info"] = {"A": structure["chain_info"]["A"]}
    return structure
