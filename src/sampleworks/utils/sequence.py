"""Utilities for applying sequence conditioning to parsed structures."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Collection, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.ccd import ChainType
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from biotite.sequence import ProteinSequence
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import get_residue_starts
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
    return sum(_heavy_atoms_per_residue(ProteinSequence.convert_letter_1to3(aa)) for aa in sequence)


_CANONICAL_AMINO_ACIDS = frozenset(
    get_1_from_3_letter_code(res_name, ChainType.POLYPEPTIDE_L) for res_name in STANDARD_AA
)


def validate_seq_with_error(sequence: str) -> str:
    """Validate a protein sequence and return it in canonical (uppercase) form.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence to validate. Case-insensitive.

    Returns
    -------
    str
        The uppercased sequence.

    Raises
    ------
    ValueError
        If *sequence* contains anything other than the 20 canonical amino acids.
    """
    sequence = sequence.upper()
    invalid = sorted(set(sequence) - _CANONICAL_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"Invalid protein sequence: non-canonical character(s) {invalid}. Only the 20 "
            f"canonical one-letter amino-acid codes are supported"
        )
    return sequence


def resolve_sequence_arg(
    value: str | os.PathLike[str] | None,
    root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve a sequence value or FASTA path to a validated amino-acid sequence.

    Parameters
    ----------
    value : str or os.PathLike or None
        An amino-acid string, a path to a FASTA file (``.fasta``, ``.fa``, ``.faa``
        or ``.fas``), or an empty value.
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

    _FASTA_EXTENSIONS = {".fasta", ".fa", ".faa", ".fas"}
    # Check the suffix first: stat() on a literal sequence > 255 chars raises ENAMETOOLONG.
    if path.suffix.lower() in _FASTA_EXTENSIONS and path.is_file():
        fasta = list(islice(FastaFile.read_iter(path), 2))
        if len(fasta) == 0:
            raise ValueError(f"FASTA file contains no sequence: {path}")
        if len(fasta) > 1:
            # TODO: Deal with ligands and multichain
            raise ValueError(f"FASTA file contains multiple sequences: {path}")
        sequence = fasta[0][1].strip()
        if not sequence:
            raise ValueError(f"FASTA file contains no sequence: {path}")

    return validate_seq_with_error(sequence)


def align_to_sequence(
    observed: Sequence[str],
    sequence: Sequence[str | Collection[str]],
    residue_ids: np.ndarray | None,
    insertion_codes: np.ndarray | None,
) -> tuple[np.ndarray, int | None]:
    """Place observed residues on a full sequence by name and numbering, optimally.

    Observed residues keep their order and each takes one sequence position. The placement
    matches the most names first, then has the fewest gaps that disagree with the numbering: a
    gap between two observed residues is free when its length is their residue-number jump
    (both without insertion codes, numbers increasing), and costs one otherwise, as does an
    adjacent pair the numbering says is apart. With numbering, gaps before the first and after
    the last residue are free (unresolved termini). Without it, every run of skipped positions
    costs one, terminal or not, so a gap inside a run of identical residues has several optimal
    places, reported as ambiguous.

    Parameters
    ----------
    observed : Sequence[str]
        Observed residues in residue order: one-letter codes, or residue names.
    sequence : Sequence[str | Collection[str]]
        Full sequence, in the same alphabet as ``observed``. A collection lists the monomers
        of a microheterogeneity position, any of which matches.
    residue_ids : np.ndarray | None
        ``res_id`` of each observed residue, shape ``(n_observed,)``, or ``None`` when the
        numbering says nothing about sequence distance.
    insertion_codes : np.ndarray | None
        ``ins_code`` of each observed residue, shape ``(n_observed,)``; required with
        ``residue_ids``.

    Returns
    -------
    tuple[np.ndarray, int | None]
        Position in ``sequence`` of each observed residue, ``-1`` where its name matches no
        position it can take, shape ``(n_observed,)``; and the smallest sequence position at
        which two optimal placements disagree, ``None`` when the optimum is unique.
    """
    n_observed, n_sequence = len(observed), len(sequence)
    positions = np.full(n_observed, -1, dtype=np.int64)
    if n_observed == 0 or n_observed > n_sequence:
        return positions, None
    columns: dict[str, list[int]] = {}
    for position, names in enumerate(sequence):
        for name in {names} if isinstance(names, str) else set(names):
            columns.setdefault(name, []).append(position)
    matches = np.zeros((n_observed, n_sequence), dtype=bool)
    for index, name in enumerate(observed):
        matches[index, columns.get(name, [])] = True

    gap = 1
    mismatch = n_observed + 2  # dearer than every gap together: names are matched first
    unreachable = np.iinfo(np.int64).max // 4
    # With numbering, unresolved termini are expected; without it, every skipped run counts.
    terminal = 0 if residue_ids is not None else gap
    leading = np.where(np.arange(n_sequence) > 0, terminal, 0)  # (n_sequence,)
    trailing = leading[::-1]  # (n_sequence,)
    # Residue-number jump to each observed residue from the one before, -1 where unknown.
    jumps = np.full(n_observed, -1, dtype=np.int64)
    if residue_ids is not None and insertion_codes is not None:
        steps = np.diff(residue_ids.astype(np.int64))
        numbered = (insertion_codes[1:] == "") & (insertion_codes[:-1] == "") & (steps > 0)
        jumps[1:] = np.where(numbered, steps - 1, -1)

    costs = np.empty((n_observed, n_sequence), dtype=np.int64)  # best placement ending here
    costs[0] = np.where(matches[0], 0, mismatch) + leading
    for index in range(1, n_observed):
        previous, jump = costs[index - 1], jumps[index]
        candidates = np.full((3, n_sequence), unreachable, dtype=np.int64)
        candidates[0, 1:] = previous[:-1] + (gap if jump > 0 else 0)  # adjacent
        if 0 < jump < n_sequence - 1:
            candidates[1, jump + 1 :] = previous[: n_sequence - jump - 1]  # gap = jump
        candidates[2, 2:] = np.minimum.accumulate(previous)[:-2] + gap  # any other gap
        best = candidates.min(axis=0)
        costs[index] = np.where(
            best >= unreachable, unreachable, best + np.where(matches[index], 0, mismatch)
        )
    totals = np.minimum(costs[-1] + trailing, unreachable)  # (n_sequence,)
    if totals.min() >= unreachable:
        return positions, None

    def trace(pick: Callable[[np.ndarray], int]) -> np.ndarray:
        """Trace one optimal placement back, choosing among tied predecessors with ``pick``."""
        path = np.empty(n_observed, dtype=np.int64)
        path[-1] = pick(np.flatnonzero(totals == totals.min()))
        for index in range(n_observed - 1, 0, -1):
            here, previous, jump = path[index], costs[index - 1], jumps[index]
            target = costs[index, here] - (0 if matches[index, here] else mismatch)
            tied = list(np.flatnonzero(previous[: max(here - 1, 0)] + gap == target))
            if here >= 1 and previous[here - 1] + (gap if jump > 0 else 0) == target:
                tied.append(here - 1)
            if jump > 0 and here - jump - 1 >= 0 and previous[here - jump - 1] == target:
                tied.append(here - jump - 1)
            path[index - 1] = pick(np.array(tied))
        return path

    lowest, highest = trace(np.min), trace(np.max)
    differ = np.flatnonzero(lowest != highest)
    ambiguous_at = int(min(lowest[differ[0]], highest[differ[0]])) if differ.size else None
    matched = matches[np.arange(n_observed), lowest]
    positions[matched] = lowest[matched]
    return positions, ambiguous_at


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
        chains, has multiple protein chains, contains invalid amino-acid
        characters, or if the protein chain looks malformed (non-amino-acid residues,
        residue numbering that contradicts the sequence, or residues that cannot be
        aligned to the override).
    """
    if sequence is None or not sequence.strip():
        return structure

    sequence = validate_seq_with_error(sequence.strip())

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
    arr = structure["asym_unit"]
    chain_id_str = protein_chain_ids[0]
    chain_ids = np.asarray(arr.chain_id)
    res_ids = np.asarray(arr.res_id)
    chain_mask = chain_ids == chain_id_str

    # Build observed 1-letter sequence in residue order.
    chain_array = arr[..., chain_mask]
    starts = get_residue_starts(chain_array, add_exclusive_stop=True)
    observed_seq = "".join(
        get_1_from_3_letter_code(n, ChainType.POLYPEPTIDE_L, use_closest_canonical=True)
        for n in chain_array.res_name[starts[:-1]]
    )
    non_amino_acids = [
        (int(res_id), str(res_name))
        for letter, res_id, res_name in zip(
            observed_seq, chain_array.res_id[starts[:-1]], chain_array.res_name[starts[:-1]]
        )
        if letter == "X"
    ]
    if non_amino_acids:
        raise ValueError(
            f"Sequence override failed: protein chain {chain_id_str} contains non-amino-acid "
            f"residues {non_amino_acids[:10]}. The CIF may be malformed (non-polymer residues "
            f"in a protein chain)."
        )

    residue_ids = chain_array.res_id[starts[:-1]]
    if np.any(np.diff(residue_ids) < 0):
        raise ValueError(
            f"Sequence override failed: residue numbering of protein chain {chain_id_str} "
            f"restarts. The CIF may be malformed."
        )

    # Map each observed residue to its position in the full (potentially longer) override.
    residue_seq_idx, _ = align_to_sequence(
        observed_seq, sequence, residue_ids, chain_array.ins_code[starts[:-1]]
    )

    # Every observed residue must map to the override sequence.  Unmapped
    # residues (seq_idx == -1) would corrupt downstream tensor indexing in
    # model wrappers (Protpardelle, RF3) that use seq_idx as array indices.
    unmapped = chain_array.res_id[starts[:-1]][residue_seq_idx < 0].tolist()
    if unmapped:
        raise ValueError(
            f"Sequence override failed: {len(unmapped)} observed residue(s) could not be "
            f"aligned to the override sequence (first 10 unmapped: {unmapped[:10]}). The override "
            f"sequence must be at least as long as, and compatible with, the observed "
            f"structure sequence. Otherwise, the CIF's residue numbering or sequence may be "
            f"malformed."
        )

    # Annotate atoms with seq_idx: aligned position for the protein chain,
    # -1 elsewhere (make_normalized_atom_id falls back to dense rank for -1).
    seq_idx = np.full(len(res_ids), -1, dtype=np.int64)
    seq_idx[chain_mask] = np.repeat(residue_seq_idx, np.diff(starts))
    updated_arr = arr.copy()
    updated_arr.set_annotation("seq_idx", seq_idx)

    return {**structure, "chain_info": updated_chain_info, "asym_unit": updated_arr}
