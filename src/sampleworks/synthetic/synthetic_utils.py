"""Shared utilities for synthetic structure-factor and density generation."""

import math
import traceback
from collections import Counter
from collections.abc import Hashable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray, get_residue_starts
from loguru import logger
from reciprocalspaceship.dtypes.base import MTZDtype

from sampleworks.utils.atom_array_utils import (
    AltlocInfo,
    apply_selection,
    BLANK_ALTLOC_IDS,
    detect_altlocs,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    remove_hydrogens,
)


# How many explicit duplicates to show in an error message when converting atomarray to gemmi.
MAX_REPORTED_DUPLICATES = 3


def resolve_parallel_jobs(device: torch.device | str, n_jobs: int) -> int:
    """Choose a safe job count for synthetic calculations on a device.

    Process-based joblib workers each create an independent CUDA context. To
    avoid GPU-memory contention and out-of-memory failures, CUDA work is kept
    in the current process while CPU work retains the caller's requested
    parallelism.

    Parameters
    ----------
    device
        Device used for the synthetic calculation.
    n_jobs
        Requested joblib worker count. Negative values request multiple workers.

    Returns
    -------
    int
        ``1`` for CUDA requests that would use multiple workers; otherwise the
        requested value.
    """
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and (n_jobs < 0 or n_jobs > 1):
        logger.warning(
            f"CUDA device {resolved_device} with n_jobs={n_jobs} would create "
            "multiple CUDA contexts and risk GPU memory exhaustion; using n_jobs=1"
        )
        return 1
    return n_jobs


def resolve_mtz_column(
    dataset: rs.DataSet,
    dtype: MTZDtype,
    *,
    column: str | None = None,
) -> str:
    """Resolve the single column of the given MTZ dtype to use from ``dataset``.

    Selects the dataset's columns of ``dtype`` and either validates a caller-supplied
    ``column`` against them or returns the sole candidate, so callers need not extract the
    candidate set themselves. Error messages label the dtype via ``dtype.name`` (e.g.
    ``"SFAmplitude"``, ``"Stddev"``, ``"Phase"``).

    Parameters
    ----------
    dataset
        The reflection dataset (e.g. a parsed MTZ) whose columns are searched.
    dtype
        MTZ dtype selecting the candidate columns (e.g.
        ``rs.StructureFactorAmplitudeDtype()``, ``rs.PhaseDtype()``).
    column
        Caller-chosen column. When given, it must be a ``dtype`` column of ``dataset`` and
        is returned verbatim. When ``None``, the sole ``dtype`` column is returned.

    Returns
    -------
    str
        The resolved column name.

    Raises
    ------
    ValueError
        If ``column`` is given but is not a ``dtype`` column of ``dataset``, or ``column``
        is ``None`` and the dataset holds zero or more than one ``dtype`` column (ambiguous
        — the caller must disambiguate by passing ``column``).
    """
    dtype_name = dtype.name
    candidates = list(dataset.select_mtzdtype(dtype).columns)
    if not candidates:
        raise ValueError(f"No {dtype_name} column found in the dataset.")
    if column is not None:
        if column not in candidates:
            raise ValueError(
                f"Requested {dtype_name} column {column!r} is not among the dataset's "
                f"{dtype_name} columns: {candidates!r}."
            )
        return column
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {dtype_name} columns {candidates} found; "
            "pass an explicit column to disambiguate."
        )
    return candidates[0]


def validate_occupancy_values(occupancy_values: list[float]) -> None:
    """Raise ValueError if any value is outside [0.0, 1.0] or values don't sum to 1.0."""
    for i, v in enumerate(occupancy_values):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Occupancy value {v} at index {i} is out of range [0.0, 1.0]")
    if occupancy_values and not math.isclose(sum(occupancy_values), 1.0):
        raise ValueError(f"Occupancy values must sum to 1.0, got {sum(occupancy_values)}")


def assign_occupancies(
    atom_array: AtomArray,
    altloc_info: AltlocInfo,
    occupancy_mode: str,
    occupancy_values: list[float] | None = None,
) -> AtomArray:
    """Assign occupancy values to atoms based on their altloc membership.

    Parameters
    ----------
    atom_array
        Structure to modify
    altloc_info
        Detected altloc information from detect_altlocs()
    occupancy_mode
        Assignment mode: 'default' (no change), 'uniform' (1/n_altlocs each),
        or 'custom' (user-specified values)
    occupancy_values
        For 'custom' mode: list of occupancy values [0.0-1.0] assigned to altlocs
        in sorted order (e.g., [0.3, 0.7] assigns 0.3 to altloc 'A', 0.7 to 'B').
        If fewer values than altlocs, remaining altlocs get occupancy 0.

    Returns
    -------
    AtomArray
        Modified structure with updated occupancies

    Raises
    ------
    ValueError
        If 'custom' mode is requested but no altlocs exist, or if occupancy_values
        is None in custom mode, or if any occupancy value is outside [0.0, 1.0]
    """
    if occupancy_mode == "default":
        return atom_array

    if not altloc_info.altloc_ids:
        if occupancy_mode == "custom":
            raise ValueError(
                "Custom occupancy mode was requested, but the structure has no altlocs."
            )
        logger.warning("No altlocs detected, using default occupancies")
        return atom_array

    result = atom_array.copy()
    occupancy = result.occupancy

    if occupancy_mode == "uniform":
        n_altlocs = len(altloc_info.altloc_ids)
        uniform_occ = 1.0 / n_altlocs
        for altloc in altloc_info.altloc_ids:
            occupancy[altloc_info.atom_masks[altloc]] = uniform_occ

    elif occupancy_mode == "custom":
        if occupancy_values is None:
            raise ValueError("occupancy_values required for custom mode")
        n_altlocs = len(altloc_info.altloc_ids)
        if len(occupancy_values) > n_altlocs:
            logger.warning(
                f"Got {len(occupancy_values)} occupancy values but structure "
                f"has {n_altlocs} altlocs. Extra values will be ignored."
            )
            occupancy_values = occupancy_values[:n_altlocs]
        elif len(occupancy_values) < n_altlocs:
            logger.warning(
                f"Got {len(occupancy_values)} occupancy values but structure "
                f"has {n_altlocs} altlocs. Missing values are automatically set to 0."
            )
            occupancy_values = occupancy_values + [0.0] * (n_altlocs - len(occupancy_values))
        validate_occupancy_values(occupancy_values)
        for altloc, occupancy_value in zip(sorted(altloc_info.altloc_ids), occupancy_values):
            occupancy[altloc_info.atom_masks[altloc]] = occupancy_value

    return result


def load_structure_for_synthetic_reward(
    structure_path: Path,
    occupancy_mode: str,
    occupancy_values: list[float],
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    selection: str | None = None,
) -> AtomArray | None:
    """Load and prepare a structure for synthetic reward generation.

    Handles loading, optional atom selection, stripping of unwanted atom classes,
    and occupancy assignment. Returns None on load or selection errors (logged);
    raises ValueError on invalid occupancy_mode or occupancy assignment errors (logged
    before raising).

    Parameters
    ----------
    structure_path
        Absolute path to the structure file
    occupancy_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform'
        (1/n_altlocs each), or 'custom' (use occupancy_values)
    occupancy_values
        Occupancy values for custom mode. If non-empty and occupancy_mode is not
        'custom', occupancy_mode is overridden with a warning.
    strip_hydrogens
        If True, remove hydrogen atoms
    strip_waters
        If True, remove water molecules
    strip_ligands
        If True, keep only polymer amino-acid atoms
    selection
        Optional atom selection string. If None, the full structure is used.

    Returns
    -------
    AtomArray | None
        Prepared structure, or None if loading or selection failed
    """
    if not structure_path.exists():
        logger.error(f"Structure not found: {structure_path}")
        return None

    try:
        atom_array = load_structure_with_altlocs(structure_path)
    except Exception as e:
        logger.error(
            f"Failed to load {structure_path} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return None

    try:
        atom_array = apply_selection(atom_array, selection)
    except ValueError as e:
        logger.error(f"Selection error for {structure_path}: {e}")
        return None

    # This is currently a sort of hacky way to remove ligands by keeping only polymer atoms
    # TODO: there's probably a more robust way to do this
    atom_array = remove_hydrogens(atom_array) if strip_hydrogens else atom_array
    atom_array = remove_waters(atom_array) if strip_waters else atom_array
    atom_array = keep_polymer(keep_amino_acids(atom_array)) if strip_ligands else atom_array
    assert isinstance(atom_array, AtomArray)
    altloc_info = detect_altlocs(atom_array)
    if occupancy_values:
        if occupancy_mode != "custom":
            logger.warning(
                f"Custom occupancy values provided for {structure_path}, "
                f"but occupancy_mode is '{occupancy_mode}'. Using 'custom' mode."
            )
            occupancy_mode = "custom"
        try:
            atom_array = assign_occupancies(
                atom_array, altloc_info, occupancy_mode, occupancy_values
            )
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {structure_path}: {e}")
            raise
    elif occupancy_mode in {"uniform", "default"}:
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, occupancy_mode)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {structure_path}: {e}")
            raise
    else:
        logger.error(f"Invalid occupancy mode '{occupancy_mode}' for {structure_path}")
        raise ValueError(f"Invalid occupancy mode '{occupancy_mode}'")

    return atom_array


def _resolve_altlocs_for_gemmi(atom_array: AtomArray) -> list[str]:
    """Resolve each atom's altloc label for gemmi, and standardize the blank altloc
    to gemmi's convention ``'\\x00'``. If the atom array has no altloc_id annotation,
    all atoms default to the blank altloc.

    Parameters
    ----------
    atom_array : AtomArray
        Structure whose ``altloc_id`` annotation (if present) is resolved.

    Returns
    -------
    list of str
        Validated per-atom altloc labels in gemmi convention.

    Raises
    ------
    ValueError
        If the resolved labels create duplicate ``(atom_name, altloc)`` pairs within
        a residue.
    """
    if "altloc_id" not in atom_array.get_annotation_categories():
        altlocs = ["\x00"] * len(atom_array)
    else:
        altlocs = ["\x00" if a in BLANK_ALTLOC_IDS else a for a in atom_array.altloc_id]
    _check_no_repeated_atoms(atom_array, altlocs)
    return altlocs


def _check_keys_unique(fields: Mapping[str, Sequence[Hashable]], *, level: str) -> None:
    """Require the key gemmi uses to identify one hierarchy level to be unique.

    Parameters
    ----------
    fields : Mapping of str to Sequence of Hashable
        The fields making up the key, as field name -> that field's value for every entry,
        in the order gemmi spells the key. All value sequences must be the same length.
        The field names are reported in the error message, so a reader can map a reported
        key positionally.
    level : str
        Singular noun for what is identified (``"chain"``, ``"residue"``, ``"atom"``). Used
        in the error message.

    Raises
    ------
    ValueError
        If any key appears more than once.
    """
    field_names = f"({', '.join(fields)})"
    values = zip(*fields.values())
    # Counter keeps first-occurrence order for error message
    repeats = [f"{key!r} x{count}" for key, count in Counter(values).items() if count > 1]
    if repeats:
        shown = ", ".join(repeats[:MAX_REPORTED_DUPLICATES])
        if len(repeats) > MAX_REPORTED_DUPLICATES:
            shown += f", ... and {len(repeats) - MAX_REPORTED_DUPLICATES} more"
        raise ValueError(
            f"gemmi identifies each {level} by {field_names}, so duplicates would be "
            f"indistinguishable: {shown}."
        )


def _check_no_repeated_atoms(atom_array: AtomArray, altlocs: list[str]) -> None:
    """Require each ``(atom_name, altloc)`` pair to be unique within its residue.

    gemmi (0.6.7) identifies an atom within a residue by that pair (seqid.hpp:124-141),
    so a repeat yields two indistinguishable atoms.

    Keyed on the full ``(chain_id, res_id, atom_name, altloc)`` for informative error
    message. Assumes that each ``(chain_id, res_id)`` occupies exactly one span, which
    should have been established by ``_prepare_residue_spans``.

    Parameters
    ----------
    atom_array : AtomArray
        Structure to check.
    altlocs : list of str
        Per-atom altloc labels in gemmi convention.

    Raises
    ------
    ValueError
        If any ``(atom_name, altloc)`` pair repeats within a residue.
    """
    _check_keys_unique(
        {
            "chain_id": atom_array.chain_id.tolist(),
            "res_id": atom_array.res_id.tolist(),
            "atom_name": atom_array.atom_name.tolist(),
            "altloc": altlocs,
        },
        level="atom",
    )


def _prepare_residue_spans(atom_array: AtomArray) -> Iterator[tuple[int, int]]:
    """Validate an atom array's residue spans and return the spans for building gemmi
    Structure hierarchically.

    ``_build_gemmi_residue`` reads per-residue fields from each span's first atom, and
    the chain loop in ``atomarray_to_gemmi`` assumes contiguous chains and residues.
    This function checks both assumptions and raises an error if they are violated.

    Residues are keyed on ``(chain_id, res_id)``, the only fields identifying a residue
    that ``_build_gemmi_residue`` writes. ``ins_code`` is not among them until issue #306
    is resolved, so a span it splits off is reported as a duplicate rather than kept.

    Parameters
    ----------
    atom_array : AtomArray
        Structure to validate for conversion to gemmi. Must be non-empty.

    Returns
    -------
    Iterator of tuple of int
        One ``(start_idx, stop_idx)`` per residue, covering the atoms
        ``atom_array[start_idx:stop_idx]`` that share the same ``(chain_id, res_id)``.

    Raises
    ------
    ValueError
         If ``atom_array`` is malformed by having duplicate residues or chains, or a
         residue's atoms disagree on ``hetero``.
    """
    chain_id = atom_array.chain_id
    # prepend True because atom 0 always starts a chain
    chain_start_idx = np.flatnonzero(np.concatenate([[True], chain_id[1:] != chain_id[:-1]]))
    _check_keys_unique({"chain_id": chain_id[chain_start_idx].tolist()}, level="chain")

    residue_span_idx = get_residue_starts(atom_array, add_exclusive_stop=True)
    span_start_idx = residue_span_idx[:-1]  # (n_residues,)
    _check_keys_unique(
        {
            "chain_id": chain_id[span_start_idx].tolist(),
            "res_id": atom_array.res_id[span_start_idx].tolist(),
        },
        level="residue",
    )
    # hetero is the only field _build_gemmi_residue reads that does not split a span
    hetero = atom_array.hetero
    changed = np.flatnonzero(hetero[1:] != hetero[:-1]) + 1  # atoms differing from the previous
    stray = changed[~np.isin(changed, span_start_idx)]  # a change inside a span, not at its start
    if len(stray):
        idx = int(stray[0])
        raise ValueError(
            f"Atoms of residue (chain {chain_id[idx]!r}, res_id {atom_array.res_id[idx]}) "
            f"disagree on hetero: atom {idx - 1} has {hetero[idx - 1]!r} but atom {idx} has "
            f"{hetero[idx]!r}. atomarray_to_gemmi reads hetero from each residue's first "
            "atom, so the differing value would be silently dropped."
        )

    return pairwise(residue_span_idx.tolist())


def _build_gemmi_residue(
    atom_array: AtomArray, start_idx: int, stop_idx: int, altlocs: list[str]
) -> gemmi.Residue:
    """Build one gemmi.Residue from a contiguous span of atoms.

    Parameters
    ----------
    atom_array : AtomArray
        Structure supplying per-atom annotations.
    start_idx : int
        Inclusive atom index of the residue's first atom; per-residue fields
        (name, seqid, subchain, het_flag) are read from this atom, which assumes the
        span agrees on them -- ``_prepare_residue_spans`` enforces that upstream.
    stop_idx : int
        Exclusive atom index marking the end of the residue's atom span.
    altlocs : list of str
        Per-atom altloc labels for the whole array (gemmi convention), indexed
        globally by atom index -- not by a residue-local offset.

    Returns
    -------
    gemmi.Residue
        Residue populated with the atoms ``atom_array[start_idx:stop_idx]``.
    """
    res_id = int(atom_array.res_id[start_idx])
    residue = gemmi.Residue()
    residue.name = atom_array.res_name[start_idx]
    residue.seqid = gemmi.SeqId(str(res_id))  # writes auth_seq_id
    residue.label_seq = res_id  # writes label_seq_id, important for saving mmCIF
    # writes label_asym_id; nothing else assigns it, since atomarray_to_gemmi
    # deliberately skips setup_entities(). Must stay single-char -- SFcalculator's
    # PDB-header step rejects the multi-char subchain ids setup_entities() invents.
    residue.subchain = atom_array.chain_id[start_idx]
    # biotite's bool `hetero` -> gemmi's single-char het_flag ('H' HETATM / 'A' ATOM)
    residue.het_flag = "H" if bool(atom_array.hetero[start_idx]) else "A"
    for atom_idx in range(start_idx, stop_idx):
        atom = gemmi.Atom()
        atom.name = atom_array.atom_name[atom_idx]
        atom.element = gemmi.Element(atom_array.element[atom_idx])
        atom.pos = gemmi.Position(*atom_array.coord[atom_idx])  # (3,) float
        atom.b_iso = float(atom_array.b_factor[atom_idx])
        atom.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)  # biotite stores no anisotropic B
        atom.occ = float(atom_array.occupancy[atom_idx])
        atom.altloc = altlocs[atom_idx]
        residue.add_atom(atom)
    return residue


def atomarray_to_gemmi(
    atom_array: AtomArray,
    unit_cell: gemmi.UnitCell | None = None,
    space_group: str | None = None,
) -> gemmi.Structure:
    """Convert a biotite AtomArray to a gemmi.Structure for SFcalculator.

    Anisotropic B-factors are set to zero since biotite does not store them.
    Blank altloc labels are converted from biotite's '' to gemmi's '\\x00'. If
    the atom array has no ``altloc_id`` annotation (e.g. arrays reconstructed by
    a model wrapper), all altlocs default to blank.

    No entities are assigned, so a cif written from the result has no entity block
    and ``_atom_site.label_entity_id`` is ``.``. Sequences are then inferred from
    ``_atom_site`` on reload, which is what model wrappers need; the cost is that
    ``chain_info`` loses ``rcsb_entity`` (only Protenix reads it, and it falls back
    to the chain id).

    Parameters
    ----------
    atom_array
        Input structure with occupancy and b_factor annotations
    unit_cell
        Crystallographic unit cell for the structure. If None, gemmi defaults
        to (1.0, 1.0, 1.0, 90.0, 90.0, 90.0) in units of Angstroms and degrees.
    space_group
        Space group (in Hermann-Mauguin string format) for the structure. If
        empty or invalid, SFcalculator defaults to P1.

    Returns
    -------
    gemmi.Structure
        Structure ready to be wrapped by SFC_Torch.io.PDBParser
    """
    if len(atom_array) == 0:
        raise ValueError("Cannot convert an empty AtomArray to a gemmi.Structure.")

    residue_spans = _prepare_residue_spans(atom_array)
    altlocs = _resolve_altlocs_for_gemmi(atom_array)

    # Group atoms into residues up front, then walk residues into chains. Validated,
    # contiguous grouping guarantees the hierarchy is well-formed by construction.
    model = gemmi.Model("1")  # numeric name -> valid mmCIF pdbx_PDB_model_num
    current_chain: gemmi.Chain | None = None
    for start_idx, stop_idx in residue_spans:
        chain_id = atom_array.chain_id[start_idx]
        if current_chain is None or chain_id != current_chain.name:
            if current_chain is not None:
                model.add_chain(current_chain)
            current_chain = gemmi.Chain(chain_id)
        current_chain.add_residue(_build_gemmi_residue(atom_array, start_idx, stop_idx, altlocs))
    if current_chain is not None:  # flush the trailing chain
        model.add_chain(current_chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    # No setup_entities(): it fabricates entities with an empty full_sequence, so the
    # written cif carries _entity/_entity_poly but no _entity_poly_seq. Atomworks reads
    # this partial block which leads to KeyError when model wrapper accessing fields like
    # `processed_entity_canonical_sequence`. When there are no entities, gemmi writes no
    # entity block and atomworks infers the sequence from _atom_site instead.
    if unit_cell is not None:
        structure.cell = unit_cell
    if space_group is not None:
        structure.spacegroup_hm = space_group
    return structure
