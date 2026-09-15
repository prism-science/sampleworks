"""Utilities for applying sequence conditioning to parsed structures."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.structure.info import residue as ccd_residue


@functools.cache
def _heavy_atoms_per_residue(three_letter: str) -> int:
    """Heavy atom count for a standard residue, excluding OXT."""
    res = ccd_residue(three_letter)
    return sum(1 for e, n in zip(res.element, res.atom_name) if e != "H" and n != "OXT")


def expected_heavy_atom_count(sequence: str) -> int:
    """Total heavy atoms (non-H, non-OXT) implied by a protein sequence.

    Parameters
    ----------
    sequence : str
        One-letter amino-acid sequence.

    Returns
    -------
    int
        Sum of heavy atoms across all residues, using CCD definitions.
    """
    return sum(
        _heavy_atoms_per_residue(ProteinSequence.convert_letter_1to3(aa)) for aa in sequence
    )


def validate_seq_with_error(sequence: str) -> None:
    """Validate that a string contains a valid protein sequence.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence to validate.

    Raises
    ------
    ValueError
        If *sequence* contains invalid amino-acid characters.
    """
    try:
        ProteinSequence(sequence)
    except Exception as e:
        raise ValueError(f"Invalid protein sequence: {e}") from e


def resolve_sequence_arg(
    value: str | os.PathLike[str] | None,
    root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve a sequence value or FASTA path to a validated amino-acid sequence.

    Parameters
    ----------
    value : str or os.PathLike or None
        An amino-acid string, a path to a FASTA file, or an empty value.
    root : str or os.PathLike or None
        Base directory for resolving relative FASTA paths.

    Returns
    -------
    str or None
        The amino-acid sequence, or ``None`` when *value* is empty.

    Raises
    ------
    ValueError
        If a FASTA file contains no sequence, more than one sequence, or the
        resolved string is not a valid protein sequence.
    """
    if value is None:
        return None

    sequence = os.fspath(value).strip()
    if not sequence:
        return None

    path = Path(sequence).expanduser()
    if root is not None and not path.is_absolute():
        path = Path(root).expanduser() / path

    if path.is_file():
        sequence_lines: list[str] = []
        record_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                record_count += 1
                if record_count > 1:
                    # TODO: Deal with ligands and multichain
                    raise ValueError(f"FASTA file contains multiple sequences: {path}")
                continue
            if line.startswith(";") and not sequence_lines:
                continue
            sequence_lines.append("".join(line.split()))

        if not sequence_lines:
            raise ValueError(f"FASTA file contains no sequence: {path}")
        sequence = "".join(sequence_lines)

    validate_seq_with_error(sequence)

    return sequence


def apply_sequence_override(structure: dict[str, Any], sequence: str | None) -> dict[str, Any]:
    """Return a structure with its protein-chain sequence overridden when provided.

    Parameters
    ----------
    structure : dict[str, Any]
        Atomworks-parsed structure containing a ``chain_info`` mapping.
    sequence : str or None
        One-letter amino-acid sequence to use for the protein chains. ``None`` or
        an empty string leaves the structure unchanged.

    Returns
    -------
    dict[str, Any]
        A shallow copy of ``structure`` with copied ``chain_info`` entries. The
        original structure and its chain metadata are not changed.

    Raises
    ------
    ValueError
        If a non-empty sequence is supplied but the structure has no protein
        chains, has multiple protein chains, or contains invalid amino-acid
        characters.
    """
    if sequence is None or not sequence.strip():
        return structure

    sequence = sequence.strip()

    validate_seq_with_error(sequence)

    chain_info = structure.get("chain_info")
    if not chain_info:
        raise ValueError("A sequence override requires structure chain_info")

    protein_chain_ids = [
        chain_id for chain_id, info in chain_info.items() if info["chain_type"].is_protein()
    ]
    if not protein_chain_ids:
        raise ValueError("A sequence override requires at least one protein chain")

    if len(protein_chain_ids) != 1:
        raise ValueError(
            f"Sequence overrides are currently supported only for single-protein-chain "
            f"structures, but this structure has {len(protein_chain_ids)} protein chains "
            f"({', '.join(protein_chain_ids)}). Pass per-chain sequences once multi-chain "
            f"support is implemented."
        )

    updated_chain_info = {chain_id: dict(info) for chain_id, info in chain_info.items()}
    updated_chain_info[protein_chain_ids[0]]["processed_entity_canonical_sequence"] = sequence

    # Align observed residues to override sequence so the AtomReconciler can
    # pair atoms correctly when missing residues are present.
    arr = structure.get("asym_unit")
    if arr is None:
        return {**structure, "chain_info": updated_chain_info}

    chain_id_str = protein_chain_ids[0]
    chain_ids = np.asarray(arr.chain_id)
    res_ids = np.asarray(arr.res_id)
    res_names = np.asarray(arr.res_name)
    chain_mask = chain_ids == chain_id_str

    # Build observed 1-letter sequence in residue order.
    chain_res_ids = res_ids[chain_mask]
    chain_res_names = res_names[chain_mask]
    _, first_idx = np.unique(chain_res_ids, return_index=True)
    ordered_idx = np.sort(first_idx)
    unique_rids = chain_res_ids[ordered_idx]
    observed_seq = "".join(
        ProteinSequence.convert_letter_3to1(n) for n in chain_res_names[ordered_idx]
    )

    # Align observed → override to map each observed residue to its position
    # in the full (potentially longer) override sequence.
    obs_prot = ProteinSequence(observed_seq)
    full_prot = ProteinSequence(sequence)
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignments = align_optimal(obs_prot, full_prot, matrix, terminal_penalty=False)
    trace = alignments[0].trace

    rid_to_seq_idx: dict[int, int] = {}
    for row in trace:
        obs_i, full_i = int(row[0]), int(row[1])
        if obs_i != -1 and full_i != -1:
            rid_to_seq_idx[int(unique_rids[obs_i])] = full_i

    # Annotate atoms with seq_idx: aligned position for the protein chain,
    # -1 elsewhere (make_normalized_atom_id falls back to dense rank for -1).
    seq_idx = np.full(len(res_ids), -1, dtype=np.int64)
    seq_idx[chain_mask] = np.array(
        [rid_to_seq_idx.get(int(r), -1) for r in chain_res_ids], dtype=np.int64
    )
    updated_arr = arr.copy()
    updated_arr.set_annotation("seq_idx", seq_idx)

    return {**structure, "chain_info": updated_chain_info, "asym_unit": updated_arr}
