"""Utilities for applying sequence conditioning to parsed structures."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.ccd import ChainType
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts
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


def _snap_runs_to_numbering(
    observed_seq: str,
    sequence: str,
    residue_ids: np.ndarray,
    seq_idx: np.ndarray,
) -> np.ndarray:
    """Place runs of consecutive residue numbers on consecutive sequence positions.

    A sequence-only alignment can tie when a residue next to a gap also occurs on the far
    side of the gap. In 5I09, for example, observed ``LYS 125 | SER 131 ARG 132 ...`` aligns
    SER 131 directly after LYS 125 as readily as after the gap. Treat each run of consecutive
    ``res_id`` values as a block: one offset (``res_id - sequence index``) places every residue
    in that run. For example, if K is residue 125 and a consecutive
    ``S R G`` run is numbered 131–133, the shared numbering offset places that run six sequence
    indices after K (offset 6). This can distinguish it from an earlier ``S R G`` that sequence
    alignment might choose when a gap makes both placements plausible. Try the best-supported
    chain-wide offset first, then the run's aligned offsets from most to least frequent, and accept
    only a placement where every residue matches its sequence letter.

    Parameters
    ----------
    observed_seq : str
        One-letter sequence of the observed residues, in residue order.
    sequence : str
        Full override sequence.
    residue_ids : np.ndarray
        ``res_id`` of each observed residue, shape ``(n_observed,)``.
    seq_idx : np.ndarray
        Aligned position in ``sequence`` of each observed residue, ``-1`` if unaligned,
        shape ``(n_observed,)``.

    Returns
    -------
    np.ndarray
        Adjusted positions, shape ``(n_observed,)``.

    Raises
    ------
    ValueError
        If the adjusted positions are not strictly increasing, which means the residue
        numbering contradicts the sequence (e.g. duplicated or reset ``res_id`` in one chain).

    Notes
    -----
    This function is a heuristic, and remains brittle. It assumes res_id steps match sequence
    positions within a run and that most residues share one chain-wide offset.
    align_optimal uses a linear -10 gap with free terminal gaps, so short runs beside a gap
    can slide, and only runs whose numbering uniquely identifies them get rescued.
    Homopolymer stretches stay ambiguous: every offset matches the letters, so a terminally-slid
    run can still be misplaced silently.
    Upgrade path: read label_seq_id / _pdbx_poly_seq_scheme when the CIF has them, or take
    a user-supplied missing-residue mask (#441). This relies on additional input or a properly
    formatted CIF.
    """

    def offsets_by_frequency(indices: np.ndarray) -> list[int]:
        aligned = indices[seq_idx[indices] >= 0]
        offsets, counts = np.unique(residue_ids[aligned] - seq_idx[aligned], return_counts=True)
        return offsets[np.argsort(-counts, kind="stable")].tolist()

    def letters_matched(offset: int) -> int:
        positions = residue_ids - offset
        return sum(
            0 <= p < len(sequence) and sequence[p] == letter
            for p, letter in zip(positions, observed_seq)
        )

    # The chain-wide offset is the one under which numbering explains the most letters.
    # alignment frequency only breaks ties (a slid alignment can tie on frequency alone).
    chain_offset = sorted(
        offsets_by_frequency(np.arange(len(residue_ids))), key=letters_matched, reverse=True
    )[:1]
    snapped = seq_idx.copy()
    run_breaks = np.flatnonzero(np.diff(residue_ids) != 1) + 1
    for run in np.split(np.arange(len(residue_ids)), run_breaks):
        for offset in chain_offset + offsets_by_frequency(run):
            candidate = residue_ids[run] - offset
            in_range = candidate.min() >= 0 and candidate.max() < len(sequence)
            if in_range and all(sequence[c] == observed_seq[i] for i, c in zip(run, candidate)):
                snapped[run] = candidate
                break

    placed = snapped[snapped >= 0]
    if not np.all(np.diff(placed) > 0):
        raise ValueError(
            "Sequence override failed: residue numbering is not consistent with the override "
            "sequence (e.g. duplicated or reset res_ids in one chain). The CIF may be malformed."
        )
    return snapped


def _align_chain_to_sequence(
    chain_array: AtomArray | AtomArrayStack, chain_id: str, sequence: str
) -> np.ndarray:
    """Map each atom of one protein chain to its residue's position in ``sequence``.

    Parameters
    ----------
    chain_array : AtomArray or AtomArrayStack
        Atoms of a single protein chain in residue order, ``n_chain_atoms`` per model.
    chain_id : str
        Chain identifier, used only in error messages.
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
        contradicts the sequence, or a residue cannot be aligned to the sequence.
    """
    starts = get_residue_starts(chain_array, add_exclusive_stop=True)
    residue_names = chain_array.res_name[starts[:-1]]
    residue_ids = chain_array.res_id[starts[:-1]]
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

    # Align observed → override to map each observed residue to its position
    # in the full (potentially longer) override sequence.
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignments = align_optimal(
        ProteinSequence(observed_seq),
        ProteinSequence(sequence),
        matrix,
        terminal_penalty=False,
        max_number=1,
    )

    # -1 marks an observed residue with no identical partner in the override.
    residue_seq_idx = np.full(len(residue_ids), -1, dtype=np.int64)
    for obs_i, full_i in alignments[0].trace:
        if obs_i != -1 and full_i != -1 and observed_seq[obs_i] == sequence[full_i]:
            residue_seq_idx[obs_i] = full_i
    residue_seq_idx = _snap_runs_to_numbering(observed_seq, sequence, residue_ids, residue_seq_idx)

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
        numbering that contradicts the sequence, or residues that cannot be
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
