"""Tests for the grid-search protein name parsers in ``sampleworks.eval.occupancy_utils``."""

import pytest
from sampleworks.eval.occupancy_utils import (
    extract_protein_and_occupancy,
    rcsb_id_from_protein_name,
)


@pytest.mark.parametrize(
    ("protein_name", "expected"),
    [
        ("3T94", "3T94"),
        ("3T94_1.0occA", "3T94"),
        ("1VME_0.25occA_0.75occB", "1VME"),
        ("9BN8_1occB", "9BN8"),
        ("1abc_0.5OCCa", "1abc"),
        ("pdb_00003t94_1.0occA", "pdb_00003t94"),
    ],
)
def test_rcsb_id_from_protein_name_returns_id(protein_name: str, expected: str) -> None:
    """An RCSB id followed only by occupancy tokens resolves to the id, case kept."""
    assert rcsb_id_from_protein_name(protein_name) == expected


@pytest.mark.parametrize(
    "protein_name",
    [
        "",
        "TEST_1.0occA",  # letter-led, not a legacy id
        "lysozyme_1.0occA",
        "4hhb_final",  # id prefix with a non-occupancy suffix
        "4hhbEXTRA",
        "1VME_single_001_0.5occA",
        "1abc_pdb_1000abcd",  # two id-like parts
        "3T94_1.0occ",  # occupancy token without an altloc label
        "3T94_1.0occA_extra",
    ],
)
def test_rcsb_id_from_protein_name_rejects_other_names(protein_name: str) -> None:
    """Names that are not an RCSB id plus occupancy tokens resolve to None."""
    assert rcsb_id_from_protein_name(protein_name) is None


@pytest.mark.parametrize(
    ("dir_name", "expected"),
    [
        ("1vme_0.5occA_0.5occB", ("1vme", {"A": 0.5, "B": 0.5})),
        ("6B8X_1.0OCCa", ("6b8x", {"A": 1.0})),
        ("1abc", ("1abc", {})),
    ],
)
def test_extract_protein_and_occupancy(
    dir_name: str, expected: tuple[str, dict[str, float]]
) -> None:
    """Occupancy tokens are parsed case-insensitively and the protein is lower-cased."""
    assert extract_protein_and_occupancy(dir_name) == expected
