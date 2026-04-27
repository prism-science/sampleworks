"""Utilities for applying sequence conditioning to parsed structures."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.ccd import ChainType
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from biotite.sequence import Alphabet, GeneralSequence, ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts
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


# Separates per-chain sequences in a multi-chain override, as in ColabFold's ``SEQ:SEQ`` input.
CHAIN_SEQUENCE_SEPARATOR = ":"


def split_chain_sequences(sequence: str) -> tuple[str, ...]:
    """Split a ``SEQ:SEQ`` override into validated per-chain sequences.

    Parameters
    ----------
    sequence : str
        One sequence per protein chain, joined by ``CHAIN_SEQUENCE_SEPARATOR``.
        A sequence without a separator is a single chain. Case-insensitive.

    Returns
    -------
    tuple[str, ...]
        Uppercased sequences, one per chain, in the order given.

    Raises
    ------
    ValueError
        If any chain sequence is empty or contains a non-canonical amino acid.
    """
    chain_sequences = tuple(part.strip() for part in sequence.split(CHAIN_SEQUENCE_SEPARATOR))
    if not all(chain_sequences):
        raise ValueError(
            f"Invalid protein sequence {sequence!r}: a chain sequence is empty. Give one "
            f"non-empty sequence per protein chain, separated by "
            f"{CHAIN_SEQUENCE_SEPARATOR!r}."
        )
    return tuple(validate_seq_with_error(chain_sequence) for chain_sequence in chain_sequences)


def resolve_sequence_arg(
    value: str | os.PathLike[str] | None,
    root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve a sequence value or FASTA path to a validated amino-acid sequence.

    Parameters
    ----------
    value : str or os.PathLike or None
        An amino-acid string (``SEQ:SEQ`` for several protein chains), a path to a
        FASTA file (``.fasta``, ``.fa``, ``.faa`` or ``.fas``) holding one record per
        protein chain, or an empty value.
    root : str or os.PathLike or None
        Base directory for resolving relative FASTA paths.

    Returns
    -------
    str or None
        The amino-acid sequence, with chains joined by ``CHAIN_SEQUENCE_SEPARATOR``,
        or ``None`` when *value* is empty.

    Raises
    ------
    ValueError
        If a FASTA file contains no sequence, a chain sequence is empty, or a
        chain sequence is not a valid protein sequence.
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
        records = [record_sequence.strip() for _, record_sequence in FastaFile.read_iter(path)]
        if not any(records):
            raise ValueError(f"FASTA file contains no sequence: {path}")
        sequence = CHAIN_SEQUENCE_SEPARATOR.join(records)

    return CHAIN_SEQUENCE_SEPARATOR.join(split_chain_sequences(sequence))


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


def _align_chain_to_sequence(
    chain_array: AtomArray | AtomArrayStack, chain_id: str, sequence: str
) -> np.ndarray:
    """Map each atom of one protein chain to its residue's position in ``sequence``.

    Parameters
    ----------
    chain_array : AtomArray or AtomArrayStack
        Atoms of a single protein chain in residue order, ``n_chain_atoms`` per model.
    chain_id : str
        Chain identifier, used only in messages.
    sequence : str
        Validated one-letter override sequence for this chain.

    Returns
    -------
    np.ndarray
        Position in ``sequence`` of each atom's residue, shape ``(n_chain_atoms,)``.

    Raises
    ------
    ValueError
        If the chain contains non-amino-acid residues, its residue numbering
        restarts or repeats, or a residue cannot be aligned to the sequence.
    """
    starts = get_residue_starts(chain_array, add_exclusive_stop=True)
    residue_names = chain_array.res_name[starts[:-1]]
    residue_ids = chain_array.res_id[starts[:-1]]
    insertion_codes = chain_array.ins_code[starts[:-1]]
    observed_seq = "".join(
        get_1_from_3_letter_code(n, ChainType.POLYPEPTIDE_L, use_closest_canonical=True)
        for n in residue_names
    )
    non_amino_acids = [
        (int(res_id), str(res_name))
        for letter, res_id, res_name in zip(observed_seq, residue_ids, residue_names)
        if letter == "X"
    ]
    if non_amino_acids:
        raise ValueError(
            f"Sequence override failed: protein chain {chain_id} contains non-amino-acid "
            f"residues {non_amino_acids[:10]}. The CIF may be malformed (non-polymer residues "
            f"in a protein chain)."
        )

    # Microheterogeneity would put one residue at two override positions.
    residue_keys = set(zip(residue_ids.tolist(), insertion_codes.tolist()))
    if np.any(np.diff(residue_ids) < 0) or len(residue_keys) < len(residue_ids):
        raise ValueError(
            f"Sequence override failed: residue numbering of protein chain {chain_id} "
            f"restarts or repeats. The CIF may be malformed."
        )

    residue_seq_idx, ambiguous_at = align_to_sequence(
        observed_seq, sequence, residue_ids, insertion_codes
    )
    if ambiguous_at is not None:
        logger.warning(
            f"Protein chain {chain_id} fits the override sequence in several equally good "
            f"ways around override position {ambiguous_at} (a gap inside a run of repeated "
            "residues that the residue numbering does not place). Using one of them."
        )

    # Unmapped residues would corrupt downstream tensor indexing in model wrappers
    # (Protpardelle, RF3) that use seq_idx as array indices.
    unmapped = residue_ids[residue_seq_idx < 0].tolist()
    if unmapped:
        raise ValueError(
            f"Sequence override failed: {len(unmapped)} observed residue(s) of protein chain "
            f"{chain_id} could not be aligned to the override sequence (first 10 unmapped: "
            f"{unmapped[:10]}). The override sequence must be at least as long as, and "
            f"compatible with, the observed structure sequence. Otherwise, the CIF's residue "
            f"numbering or sequence may be malformed."
        )
    return np.repeat(residue_seq_idx, np.diff(starts))


def _resolved_residue_count(atom_array: AtomArray | AtomArrayStack, chain_id: str) -> int:
    """Number of residues with coordinates in one chain of ``atom_array``."""
    return len(get_residue_starts(atom_array[..., np.asarray(atom_array.chain_id) == chain_id]))


def infer_sequence_override(structure: dict[str, Any]) -> str | None:
    """Return the file's own sequence for a protein chain with unresolved residues.

    Every model wrapper generates the full ``processed_entity_canonical_sequence`` of each
    chain, so when the structure file states a sequence longer than its resolved residues,
    the input and model atom arrays only pair correctly after
    :func:`apply_sequence_override` places each resolved residue in that sequence.

    Parameters
    ----------
    structure : dict[str, Any]
        Atomworks-parsed structure with ``"asym_unit"`` and ``"chain_info"``.

    Returns
    -------
    str or None
        The stated sequence of the single protein chain when it has unresolved residues;
        ``None`` when no protein chain has unresolved residues, the file states no sequence
        (atomworks then reads it off the coordinates), or there are several protein chains,
        which this inference does not handle yet.
    """
    protein_chain_info = {
        chain_id: info
        for chain_id, info in (structure.get("chain_info") or {}).items()
        if info["chain_type"].is_protein()
    }
    atom_array = structure["asym_unit"]
    # atomworks sets "unprocessed_entity_canonical_sequence" only for sequences it read from
    # the entity_poly category rather than from the resolved residues.
    gapped_chain_ids = [
        chain_id
        for chain_id, info in protein_chain_info.items()
        if "unprocessed_entity_canonical_sequence" in info
        and len(info["processed_entity_canonical_sequence"])
        > _resolved_residue_count(atom_array, chain_id)
    ]
    if not gapped_chain_ids:
        return None
    if len(protein_chain_info) != 1:
        logger.warning(
            f"Protein chains {gapped_chain_ids} have unresolved residues, but sequences can only "
            f"be inferred for single-protein-chain structures; model atoms in these chains may "
            f"be paired with the wrong input residues."
        )
        return None

    logger.info(
        f"Protein chain {gapped_chain_ids[0]} has unresolved residues; using the sequence stated "
        f"in the structure file as the sequence override. Pass --sequence to choose another."
    )
    return protein_chain_info[gapped_chain_ids[0]]["processed_entity_canonical_sequence"]


def apply_sequence_override(structure: dict[str, Any], sequence: str | None) -> dict[str, Any]:
    """Return a structure with its protein-chain sequences overridden when provided.

    Parameters
    ----------
    structure : dict[str, Any]
        Atomworks-parsed structure containing a ``chain_info`` mapping.
    sequence : str or None
        One-letter amino-acid sequence per protein chain, joined by
        ``CHAIN_SEQUENCE_SEPARATOR`` (``SEQ:SEQ``) in the order the protein chains
        appear in ``chain_info``. Non-protein chains (ligands, nucleic acids) keep
        their parsed definitions. ``None`` or an empty string leaves the structure
        unchanged.

    Returns
    -------
    dict[str, Any]
        A shallow copy of ``structure`` with copied ``chain_info`` entries and a
        copied ``asym_unit`` carrying a ``seq_idx`` annotation: each atom's position
        in its chain's override sequence, ``-1`` for atoms of non-protein chains.
        The original structure and its chain metadata are not changed.

    Raises
    ------
    ValueError
        If a non-empty sequence is supplied but the structure has no protein
        chains, the number of chain sequences differs from the number of protein
        chains, a chain sequence is empty or contains invalid amino-acid characters,
        or a protein chain looks malformed (non-amino-acid residues, residue
        numbering that restarts or repeats, or residues that cannot be
        aligned to the override).
    """
    if sequence is None or not sequence.strip():
        return structure

    chain_sequences = split_chain_sequences(sequence)

    chain_info = structure.get("chain_info")
    if not chain_info:
        raise ValueError("A sequence override requires structure chain_info")

    protein_chain_ids = [
        chain_id for chain_id, info in chain_info.items() if info["chain_type"].is_protein()
    ]
    if not protein_chain_ids:
        raise ValueError("A sequence override requires at least one protein chain")

    if len(chain_sequences) != len(protein_chain_ids):
        raise ValueError(
            f"The sequence override has {len(chain_sequences)} chain sequence(s), but this "
            f"structure has {len(protein_chain_ids)} protein chains "
            f"({', '.join(protein_chain_ids)}). Give one sequence per protein chain, in that "
            f"order, separated by {CHAIN_SEQUENCE_SEPARATOR!r}."
        )

    arr = structure["asym_unit"]
    chain_ids = np.asarray(arr.chain_id)
    # -1 marks atoms outside the overridden protein chains; make_normalized_atom_id
    # falls back to dense rank for them.
    seq_idx = np.full(arr.array_length(), -1, dtype=np.int64)
    updated_chain_info = {chain_id: dict(info) for chain_id, info in chain_info.items()}
    for chain_id, chain_sequence in zip(protein_chain_ids, chain_sequences, strict=True):
        chain_mask = chain_ids == chain_id
        seq_idx[chain_mask] = _align_chain_to_sequence(
            arr[..., chain_mask], chain_id, chain_sequence
        )
        updated_chain_info[chain_id]["processed_entity_canonical_sequence"] = chain_sequence

    updated_arr = arr.copy()
    updated_arr.set_annotation("seq_idx", seq_idx)

    return {**structure, "chain_info": updated_chain_info, "asym_unit": updated_arr}


def _chain_sequence_positions(
    chain_id: str, residue_ids: np.ndarray, info: dict[str, Any]
) -> np.ndarray:
    """Position of each observed residue of one polymer chain in its deposited sequence.

    Parameters
    ----------
    chain_id : str
        Chain being placed, used in error messages.
    residue_ids : np.ndarray
        ``res_id`` of each observed residue in order, shape ``(n_residues,)``.
    info : dict[str, Any]
        The chain's ``chain_info`` entry.

    Returns
    -------
    np.ndarray
        0-based sequence position of each residue, shape ``(n_residues,)``.

    Raises
    ------
    ValueError
        If the chain is not fully observed and its residues cannot be placed through
        ``info["res_id"]``.
    """
    sequence_length = len(info["processed_entity_canonical_sequence"])
    # A fully observed chain needs no numbering, which also keeps insertion-code residues apart.
    if len(residue_ids) == sequence_length:
        return np.arange(len(residue_ids), dtype=np.int64)
    if "res_id" not in info:
        raise ValueError(
            f"Chain {chain_id} has {len(residue_ids)} observed residues but a sequence of "
            f"{sequence_length}, and its chain_info has no res_id numbering to place them"
        )
    position_of = {int(res_id): i for i, res_id in enumerate(info["res_id"])}
    unplaced = sorted({int(res_id) for res_id in residue_ids} - position_of.keys())
    if unplaced:
        raise ValueError(
            f"Chain {chain_id}: res_id(s) {unplaced[:10]} are not in the deposited sequence "
            f"numbering (chain_info res_id)"
        )
    return np.array([position_of[int(res_id)] for res_id in residue_ids], dtype=np.int64)


def annotate_sequence_positions(atom_array: AtomArray, chain_info: dict[str, Any]) -> AtomArray:
    """Return a copy annotated with each atom's position in its chain's deposited sequence.

    Parameters
    ----------
    atom_array : AtomArray
        Atoms to annotate. Not modified.
    chain_info : dict[str, Any]
        Atomworks ``chain_info`` for the structure the atoms came from.

    Returns
    -------
    AtomArray
        Copy with an int64 ``seq_idx`` annotation: the 0-based index into
        ``processed_entity_canonical_sequence``. Chains that already carry a
        ``seq_idx >= 0`` (e.g. from ``apply_sequence_override``) keep it.

    Raises
    ------
    ValueError
        If a partially observed polymer chain has a residue whose ``res_id`` is not in
        ``chain_info[chain_id]["res_id"]``.
    """
    annotated = atom_array.copy()
    chain_ids = np.asarray(annotated.chain_id)
    if "seq_idx" in annotated.get_annotation_categories():
        seq_idx = np.asarray(annotated.seq_idx, dtype=np.int64)
    else:
        # -1: atom has no position in a deposited polymer sequence.
        seq_idx = np.full(len(chain_ids), -1, dtype=np.int64)
    preserved_chains = set(chain_ids[seq_idx >= 0])

    starts = get_residue_starts(annotated, add_exclusive_stop=True)
    residue_sizes = np.diff(starts)
    residue_chain_ids = chain_ids[starts[:-1]]
    residue_ids = np.asarray(annotated.res_id)[starts[:-1]]
    for chain_id, info in chain_info.items():
        if chain_id in preserved_chains or not info["chain_type"].is_polymer():
            continue
        in_chain = residue_chain_ids == chain_id
        if not in_chain.any():
            continue
        positions = _chain_sequence_positions(chain_id, residue_ids[in_chain], info)
        seq_idx[np.repeat(in_chain, residue_sizes)] = np.repeat(positions, residue_sizes[in_chain])

    annotated.set_annotation("seq_idx", seq_idx)
    return annotated
