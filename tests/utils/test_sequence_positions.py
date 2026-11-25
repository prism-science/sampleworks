"""Tests for annotating atoms with their position in the deposited sequence."""

from typing import Any

import numpy as np
import pytest
from atomworks.enums import ChainType
from biotite.structure import AtomArray
from sampleworks.utils.atom_array_utils import make_normalized_atom_id
from sampleworks.utils.sequence import annotate_sequence_positions


def _chain_atoms(
    chain_id: str, residues: list[tuple[int, str, str]], atom_names: tuple[str, ...] = ("N", "CA")
) -> AtomArray:
    """Build one chain with the given atoms per residue.

    Parameters
    ----------
    chain_id : str
        Chain ID for every atom.
    residues : list[tuple[int, str, str]]
        ``(res_id, ins_code, res_name)`` per residue, in order.
    atom_names : tuple[str, ...]
        Atom names given to each residue.

    Returns
    -------
    AtomArray
        ``len(residues) * len(atom_names)`` atoms.
    """
    n_per = len(atom_names)
    atoms = AtomArray(len(residues) * n_per)
    atoms.chain_id[:] = chain_id
    atoms.res_id = np.repeat([r[0] for r in residues], n_per)
    atoms.ins_code = np.repeat([r[1] for r in residues], n_per)
    atoms.res_name = np.repeat([r[2] for r in residues], n_per)
    atoms.atom_name = np.tile(atom_names, len(residues))
    atoms.coord = np.zeros((len(atoms), 3), dtype=np.float32)
    return atoms


def _protein_info(sequence: str, res_ids: list[int] | None = None) -> dict[str, Any]:
    """``chain_info`` entry for a protein chain."""
    info: dict[str, Any] = {
        "chain_type": ChainType.POLYPEPTIDE_L,
        "processed_entity_canonical_sequence": sequence,
    }
    if res_ids is not None:
        info["res_id"] = res_ids
    return info


def _residue_seq_idx(atoms: AtomArray) -> list[int]:
    """``seq_idx`` of each residue, read from its first atom (two atoms per residue)."""
    return atoms.seq_idx[::2].tolist()


def test_fully_observed_chain_uses_residue_ordinal_across_insertion_codes():
    """Residues sharing a res_id via insertion codes get distinct consecutive positions."""
    atoms = _chain_atoms("A", [(5, "", "MET"), (6, "", "ALA"), (6, "A", "CYS"), (7, "", "ASP")])

    annotated = annotate_sequence_positions(atoms, {"A": _protein_info("MACD")})

    assert annotated.seq_idx.dtype == np.int64
    assert _residue_seq_idx(annotated) == [0, 1, 2, 3]


def test_partially_observed_chain_uses_res_id_numbering():
    """An unresolved N-terminus shifts positions and an internal gap is skipped."""
    atoms = _chain_atoms("A", [(3, "", "ALA"), (4, "", "CYS"), (6, "", "GLU")])
    info = _protein_info("MKACDE", res_ids=[1, 2, 3, 4, 5, 6])

    annotated = annotate_sequence_positions(atoms, {"A": info})

    assert _residue_seq_idx(annotated) == [2, 3, 5]


def test_existing_seq_idx_is_preserved():
    """Chains already carrying seq_idx (e.g. from a sequence override) are left as they are."""
    atoms = _chain_atoms("A", [(1, "", "ALA"), (2, "", "CYS")])
    atoms.set_annotation("seq_idx", np.repeat([4, 7], 2))

    annotated = annotate_sequence_positions(atoms, {"A": _protein_info("AC")})

    assert _residue_seq_idx(annotated) == [4, 7]


def test_ligand_and_unknown_chains_get_sentinel():
    """Non-polymer chains and chains absent from chain_info are marked -1."""
    protein = _chain_atoms("A", [(1, "", "ALA")])
    ligand = _chain_atoms("B", [(1, "", "HEM")], atom_names=("FE",))
    unknown = _chain_atoms("C", [(1, "", "HOH")], atom_names=("O",))
    chain_info = {
        "A": _protein_info("A"),
        "B": {"chain_type": ChainType.NON_POLYMER, "processed_entity_canonical_sequence": ""},
    }

    annotated = annotate_sequence_positions(protein + ligand + unknown, chain_info)

    assert annotated.seq_idx.tolist() == [0, 0, -1, -1]


def test_unknown_res_id_raises():
    """A residue outside the deposited numbering is an error naming chain and res_id."""
    atoms = _chain_atoms("A", [(1, "", "ALA"), (99, "", "CYS")])

    with pytest.raises(ValueError, match=r"Chain A.*\[99\]"):
        annotate_sequence_positions(atoms, {"A": _protein_info("ACD", res_ids=[1, 2, 3])})


def test_partial_chain_without_res_id_numbering_raises():
    """A partially observed chain cannot be placed without chain_info res_id."""
    atoms = _chain_atoms("A", [(1, "", "ALA")])

    with pytest.raises(ValueError, match="Chain A"):
        annotate_sequence_positions(atoms, {"A": _protein_info("AC")})


def test_input_is_not_mutated():
    """The input keeps its annotations and the result is a separate array."""
    atoms = _chain_atoms("A", [(1, "", "ALA"), (2, "", "CYS")])
    with_seq_idx = atoms.copy()
    with_seq_idx.set_annotation("seq_idx", np.full(len(atoms), -1, dtype=np.int64))

    annotate_sequence_positions(atoms, {"A": _protein_info("AC")})
    annotate_sequence_positions(with_seq_idx, {"A": _protein_info("AC")})

    assert "seq_idx" not in atoms.get_annotation_categories()
    assert with_seq_idx.seq_idx.tolist() == [-1, -1, -1, -1]


def test_annotation_matches_atoms_across_missing_terminus():
    """A chain missing its N-terminus gets the same normalized atom IDs as the full chain."""
    residues = [(1, "", "MET"), (2, "", "LYS"), (3, "", "ALA"), (4, "", "CYS")]
    info = _protein_info("MKAC", res_ids=[1, 2, 3, 4])
    full = annotate_sequence_positions(_chain_atoms("A", residues), {"A": info})
    partial = annotate_sequence_positions(_chain_atoms("A", residues[2:]), {"A": info})

    np.testing.assert_array_equal(
        make_normalized_atom_id(partial), make_normalized_atom_id(full)[4:]
    )
