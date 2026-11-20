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
from biotite.sequence import Alphabet, GeneralSequence, ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import get_residue_starts
from biotite.structure.info import residue as ccd_residue
from loguru import logger


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


_GAP_COST = 1
_UNREACHABLE = np.iinfo(np.int64).max // 4


def _name_set(names: str | Collection[str]) -> set[str]:
    """Return a residue's names as a set.

    Parameters
    ----------
    names : str | Collection[str]
        One name, or the names observed at one position.

    Returns
    -------
    set[str]
        The names.
    """
    return {names} if isinstance(names, str) else set(names)


def _match_matrix(
    observed: Sequence[str | Collection[str]], sequence: Sequence[str | Collection[str]]
) -> np.ndarray:
    """Mark the sequence positions whose names match each observed residue.

    Parameters
    ----------
    observed : Sequence[str | Collection[str]]
        Observed residues.
    sequence : Sequence[str | Collection[str]]
        Sequence positions, in the same alphabet as ``observed``.

    Returns
    -------
    np.ndarray
        Boolean matrix, shape ``(n_observed, n_sequence)``.
    """
    columns: dict[str, list[int]] = {}
    for position, names in enumerate(sequence):
        for name in _name_set(names):
            columns.setdefault(name, []).append(position)
    matches = np.zeros((len(observed), len(sequence)), dtype=bool)
    for index, names in enumerate(observed):
        for name in _name_set(names):
            matches[index, columns.get(name, [])] = True
    return matches


def _numbering_jumps(
    residue_ids: np.ndarray | None, insertion_codes: np.ndarray | None, n_observed: int
) -> np.ndarray:
    """Count the residues the numbering skips before each observed residue.

    Parameters
    ----------
    residue_ids : np.ndarray | None
        ``res_id`` of each observed residue, shape ``(n_observed,)``.
    insertion_codes : np.ndarray | None
        ``ins_code`` of each observed residue, shape ``(n_observed,)``.
    n_observed : int
        Number of observed residues.

    Returns
    -------
    np.ndarray
        Skipped residue count, ``-1`` where the numbering gives no distance (first residue,
        insertion codes, or numbers that do not increase), shape ``(n_observed,)``.
    """
    jumps = np.full(n_observed, -1, dtype=np.int64)
    if residue_ids is not None and insertion_codes is not None:
        steps = np.diff(residue_ids.astype(np.int64))
        numbered = (insertion_codes[1:] == "") & (insertion_codes[:-1] == "") & (steps > 0)
        jumps[1:] = np.where(numbered, steps - 1, -1)
    return jumps


def _placement_costs(
    matches: np.ndarray, jumps: np.ndarray, mismatch: int, leading: np.ndarray
) -> np.ndarray:
    """Compute a cost matrix: for each ``i`` and ``p``, the cheapest cost of placing residues
    ``[0, i]`` given that residue ``i`` is placed at position ``p``.

    Parameters
    ----------
    matches : np.ndarray
        Name matches, shape ``(n_observed, n_sequence)``.
    jumps : np.ndarray
        Skipped residue counts from :func:`_numbering_jumps`, shape ``(n_observed,)``.
    mismatch : int
        Cost of placing a residue on a position whose names differ.
    leading : np.ndarray
        Cost of the gap before the first residue, by its position, shape ``(n_sequence,)``.

    Returns
    -------
    np.ndarray
        ``costs[i, p]``: cheapest placement of residues ``0..i`` with ``i`` at ``p``,
        shape ``(n_observed, n_sequence)``.
    """
    n_observed, n_sequence = matches.shape
    costs = np.empty((n_observed, n_sequence), dtype=np.int64)
    costs[0] = np.where(matches[0], 0, mismatch) + leading
    for index in range(1, n_observed):
        previous, jump = costs[index - 1], jumps[index]
        candidates = np.full((3, n_sequence), _UNREACHABLE, dtype=np.int64)
        candidates[0, 1:] = previous[:-1] + (_GAP_COST if jump > 0 else 0)  # adjacent
        if 0 < jump < n_sequence - 1:
            candidates[1, jump + 1 :] = previous[: n_sequence - jump - 1]  # gap = jump
        candidates[2, 2:] = np.minimum.accumulate(previous)[:-2] + _GAP_COST  # any other gap
        best = candidates.min(axis=0)
        costs[index] = np.where(
            best >= _UNREACHABLE, _UNREACHABLE, best + np.where(matches[index], 0, mismatch)
        )
    return costs


def _trace_placement(
    costs: np.ndarray,
    totals: np.ndarray,
    matches: np.ndarray,
    jumps: np.ndarray,
    mismatch: int,
    pick: Callable[[np.ndarray], int],
) -> np.ndarray:
    """Trace one optimal placement back from the last observed residue.

    Parameters
    ----------
    costs : np.ndarray
        From :func:`_placement_costs`, shape ``(n_observed, n_sequence)``.
    totals : np.ndarray
        Cost of each full placement by its last position, shape ``(n_sequence,)``.
    matches : np.ndarray
        Name matches, shape ``(n_observed, n_sequence)``.
    jumps : np.ndarray
        Skipped residue counts, shape ``(n_observed,)``.
    mismatch : int
        Cost of placing a residue on a position whose names differ.
    pick : Callable[[np.ndarray], int]
        Chooses one position among tied candidates (e.g. ``np.min``).

    Returns
    -------
    np.ndarray
        Sequence position of each observed residue, shape ``(n_observed,)``.
    """
    n_observed = len(costs)
    path = np.empty(n_observed, dtype=np.int64)
    path[-1] = pick(np.flatnonzero(totals == totals.min()))
    for index in range(n_observed - 1, 0, -1):
        here, previous, jump = path[index], costs[index - 1], jumps[index]
        target = costs[index, here] - (0 if matches[index, here] else mismatch)
        tied = list(np.flatnonzero(previous[: max(here - 1, 0)] + _GAP_COST == target))
        if here >= 1 and previous[here - 1] + (_GAP_COST if jump > 0 else 0) == target:
            tied.append(here - 1)
        if jump > 0 and here - jump - 1 >= 0 and previous[here - jump - 1] == target:
            tied.append(here - jump - 1)
        path[index - 1] = pick(np.array(tied))
    return path


def align_to_sequence(
    observed: Sequence[str | Collection[str]],
    sequence: Sequence[str | Collection[str]],
    residue_ids: np.ndarray | None,
    insertion_codes: np.ndarray | None,
) -> tuple[np.ndarray, int | None]:
    """Place observed residues on a full sequence by name and residue numbering.

    Each observed residue keeps its order and takes one sequence position. The optimal
    placement matches the most names, then has the fewest gaps that disagree with the
    numbering. Without numbering, a gap inside a run of identical residues has several
    optimal places, which is reported as ambiguous.

    Parameters
    ----------
    observed : Sequence[str | Collection[str]]
        Observed residues in order, as one-letter codes or residue names. A collection holds
        the names observed at one residue, any of which matches.
    sequence : Sequence[str | Collection[str]]
        Full sequence, in the same alphabet as ``observed``.
    residue_ids : np.ndarray | None
        ``res_id`` of each observed residue, shape ``(n_observed,)``, or ``None`` when the
        numbering says nothing about sequence distance.
    insertion_codes : np.ndarray | None
        ``ins_code`` of each observed residue, shape ``(n_observed,)``; required with
        ``residue_ids``.

    Returns
    -------
    tuple[np.ndarray, int | None]
        Sequence position of each observed residue, ``-1`` where no name matches, shape
        ``(n_observed,)``; and a sequence position where two optimal placements disagree,
        or ``None`` when the optimum is unique.
    """
    n_observed, n_sequence = len(observed), len(sequence)
    positions = np.full(n_observed, -1, dtype=np.int64)
    if n_observed == 0 or n_observed > n_sequence:
        return positions, None
    matches = _match_matrix(observed, sequence)
    jumps = _numbering_jumps(residue_ids, insertion_codes, n_observed)
    mismatch = n_observed + 2  # dearer than every gap together, so names are matched first
    terminal = 0 if residue_ids is not None else _GAP_COST  # numbered termini may be unresolved
    leading = np.where(np.arange(n_sequence) > 0, terminal, 0)  # (n_sequence,)
    costs = _placement_costs(matches, jumps, mismatch, leading)
    totals = np.minimum(costs[-1] + leading[::-1], _UNREACHABLE)  # (n_sequence,)

    lowest = _trace_placement(costs, totals, matches, jumps, mismatch, np.min)
    highest = _trace_placement(costs, totals, matches, jumps, mismatch, np.max)
    disagreeing = np.flatnonzero(lowest != highest)
    ambiguous_at = (
        int(min(lowest[disagreeing[0]], highest[disagreeing[0]])) if disagreeing.size else None
    )
    matched = matches[np.arange(n_observed), lowest]
    positions[matched] = lowest[matched]
    return positions, ambiguous_at


def align_sequences(
    first: Sequence[str | Collection[str]], second: Sequence[str | Collection[str]]
) -> tuple[np.ndarray, bool]:
    """Match up the residues of two sequences by residue name, allowing gaps in either.

    For example, ``["SER", "GLY", "HIS"]`` and ``["GLY", "HIS", "TRP"]`` match GLY to GLY
    and HIS to HIS, leaving SER and TRP unmatched. The result matches as many residues as
    possible, and among equally good results uses the fewest gaps. Unmatched residues at the
    start or end of a sequence do not count as gaps.

    Parameters
    ----------
    first : Sequence[str | Collection[str]]
        Residue names in chain order, e.g. ``["GLY", "ALA"]``. A position can list several
        names, e.g. ``{"ALA", "SER"}`` for a residue modeled as two residue types, and then
        matches a residue with any of those names.
    second : Sequence[str | Collection[str]]
        Residue names in chain order, given the same way as ``first``.

    Returns
    -------
    positions : np.ndarray
        For each residue of ``first``, the index of the ``second`` residue it matched, or
        ``-1`` if it matched none. Shape ``(len(first),)``.
    ambiguous : bool
        Whether a different matching is just as good, e.g. when ``second`` repeats a residue
        and either copy could be the match.
    """
    first_sets = [frozenset(_name_set(names)) for names in first]
    second_sets = [frozenset(_name_set(names)) for names in second]
    positions = np.full(len(first_sets), -1, dtype=np.int64)
    if not first_sets or not second_sets:
        return positions, False
    symbols = list(set(first_sets) | set(second_sets))
    alphabet = Alphabet(symbols)
    # one match is worth more than all possible gaps combined
    match = len(first_sets) + len(second_sets) + 1
    scores = np.array([[match if a & b else 0 for b in symbols] for a in symbols])
    alignments = align_optimal(
        GeneralSequence(alphabet, first_sets),
        GeneralSequence(alphabet, second_sets),
        SubstitutionMatrix(alphabet, alphabet, scores),
        gap_penalty=-1,
        terminal_penalty=False,
        max_number=2,
    )
    first_index, second_index = alignments[0].trace[(alignments[0].trace >= 0).all(axis=1)].T
    same_name = [bool(first_sets[i] & second_sets[j]) for i, j in zip(first_index, second_index)]
    positions[first_index[same_name]] = second_index[same_name]
    return positions, len(alignments) > 1


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
        residue numbering that restarts or repeats, or residues that cannot be
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
    insertion_codes = chain_array.ins_code[starts[:-1]]
    # Microheterogeneity would put one residue at two override positions.
    residue_keys = set(zip(residue_ids.tolist(), insertion_codes.tolist()))
    if np.any(np.diff(residue_ids) < 0) or len(residue_keys) < len(residue_ids):
        raise ValueError(
            f"Sequence override failed: residue numbering of protein chain {chain_id_str} "
            f"restarts or repeats. The CIF may be malformed."
        )

    residue_seq_idx, ambiguous_at = align_to_sequence(
        observed_seq, sequence, residue_ids, insertion_codes
    )
    if ambiguous_at is not None:
        logger.warning(
            f"Protein chain {chain_id_str} fits the override sequence in several equally good "
            f"ways around override position {ambiguous_at} (a gap inside a run of repeated "
            "residues that the residue numbering does not place). Using one of them."
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
