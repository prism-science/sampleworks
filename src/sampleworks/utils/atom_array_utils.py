import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Literal, overload

import einx
import numpy as np
from atomworks import parse
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, filter_amino_acids, filter_polymer, stack
from biotite.structure.io.pdbx import CIFFile, set_structure
from loguru import logger


BACKBONE_ATOM_TYPES = ["C", "CA", "N", "O"]
BLANK_ALTLOC_IDS = {"", ".", " ", "?"}
ATOMWORKS_COMPARISON_OPS = ("==", ">", "<", "<=", ">=", " in ")


@dataclass
class AltlocInfo:
    """Information about alternate conformations (altlocs) in a structure.

    Attributes
    ----------
    altloc_ids
        Sorted list of altloc identifiers (e.g., ['A', 'B'])
    atom_masks
        Dictionary mapping each altloc ID to a boolean mask indicating which atoms
        belong to that altloc
    """

    altloc_ids: list[str]
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]


# TODO: migrate this to atomworks's selection algebra that is added to AtomArray/Stack
#   https://github.com/prism-science/sampleworks/issues/56
def parse_selection_string(selection: str) -> tuple[str | None, int | None, int | None]:
    """Parse a selection string like 'chain A and resi 326-339'.

    Supports both residue ranges ('resi 10-50') and single residues ('resi 10').
    For single residues, resi_start and resi_end will be equal.

    Parameters
    ----------
    selection : str
        Selection string (e.g., 'chain A', 'resi 10', 'chain A and resi 10-50')

    Returns
    -------
    tuple
        (chain_id, resi_start, resi_end) - any component may be None if not specified
    """
    # Parse "chain X and resi N-M" format and generalizations of that.
    chain = re.search(r"chain\s+(\w+)", selection, re.IGNORECASE)
    if chain is not None:
        chain = chain.group(1).upper()

    residues = re.search(r"resi\s+(\d+)(?:-(\d+))?", selection, re.IGNORECASE)
    if residues is not None:
        resi_start = int(residues.group(1))
        resi_end = int(residues.group(2)) if residues.group(2) else resi_start
    else:
        resi_start = resi_end = None

    if chain is None and resi_start is None and resi_end is None:
        logger.warning(
            "Selection string did not match any known patterns (e.g. 'chain A', 'resi 10-50')"
        )

    return chain, resi_start, resi_end


def apply_selection(atom_array: AtomArray, selection: str | None) -> AtomArray:
    """Apply an atom selection string to filter a structure.

    Parameters
    ----------
    atom_array
        Structure to filter
    selection
        Selection string (e.g., 'chain A and resi 10-50'). If None, returns
        the entire structure unchanged.

    Returns
    -------
    AtomArray
        Filtered structure containing only atoms matching the selection

    Raises
    ------
    ValueError
        If the selection string matches no atoms
    """
    if selection is None:
        return atom_array

    if not any(x in selection for x in ATOMWORKS_COMPARISON_OPS):
        mask = get_mask_from_old_selection_string(atom_array, selection)
    else:
        mask = atom_array.mask(selection)

    return cast(AtomArray, atom_array[mask])


def get_mask_from_old_selection_string(
    atom_array: AtomArray | AtomArrayStack, selection: str
) -> np.ndarray[tuple[int], np.dtype[Any]]:
    """Build an atom mask from a legacy chain and residue selection string."""
    DeprecationWarning(
        f"Using old-style selection strings like {selection} is deprecated."
        f" Use atomworks/pandas style selection strings instead."
    )
    chain_id, resi_start, resi_end = parse_selection_string(selection)
    # use the length of any of the required non-coord attributes to get the mask shape
    mask = np.ones(len(atom_array.res_id), dtype=bool)

    if chain_id is not None:
        mask &= atom_array.chain_id == chain_id

    if resi_start is not None:
        res_ids = cast(np.ndarray, atom_array.res_id)
        if resi_end is not None:
            mask &= (res_ids >= resi_start) & (res_ids <= resi_end)
        else:
            mask &= res_ids == resi_start

    if mask.sum() == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms")
    return mask


def parse_structure(path: str | Path) -> dict[str, Any]:
    """Parse a structure with the options used by Sampleworks production kwargs in
    guidance.

    Parameters
    ----------
    path : str | Path
        Path to a PDB or mmCIF structure file.

    Returns
    -------
    dict[str, Any]
        Parsed Atomworks structure and metadata, including ``"asym_unit"`` and
        ``"chain_info"``.
    """
    return parse(
        Path(path),
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )


def load_structure_with_altlocs(path: Path) -> AtomArray:
    """Load a structure file with alternate conformations and occupancy data.

    Takes the first model if multiple models are present.

    Parameters
    ----------
    path
        Path to the structure file (PDB, mmCIF, etc.)

    Returns
    -------
    AtomArray
        Loaded structure with occupancy and B-factor data
    """
    # Currently, we need to specify extra_fields=["occupancy"] to load altlocs properly
    atom_array = load_any(path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(atom_array, AtomArrayStack):
        atom_array = cast(AtomArray, atom_array[0])
    return atom_array


def save_structure_to_cif(
    atom_array: AtomArray | AtomArrayStack,
    output_path: str | Path,
    handle_nan: bool = True,
    frame_indices: int | list[int] | None = None,
) -> Path:
    """
    Save an AtomArray or AtomArrayStack to CIF file using biotite.

    Supports saving single-model or multimodel CIF files. If a stack is provided
    and multiple frames are selected, each frame becomes a separate model in the
    output CIF file.

    Parameters
    ----------
    atom_array: AtomArray | AtomArrayStack
        The structure to save.
    output_path: str | Path
        Output file path. Parent directories will be created if needed.
    handle_nan: bool, default=True
        If True, filters out atoms with NaN coordinates and warns.
        If False, passes structure as-is (may cause errors if NaN present).
    frame_indices: int | list[int] | None, default=None
        Which frames to save:
        - None: Save all frames (if AtomArrayStack) or single frame (if AtomArray)
        - int: Extract single frame at this index (saves single-model CIF)
        - list[int]: Extract multiple frames (saves multimodel CIF)

    Returns
    -------
    Path
        The path to the saved CIF file.

    Examples
    --------
    >>> from atomworks.io import parse
    >>> structure = parse("structure.pdb", ccd_mirror_path=None)
    >>> atom_array = structure["asym_unit"]
    >>>
    >>> save_structure_to_cif(atom_array, "output.cif")
    PosixPath('output.cif')
    >>>
    >>> save_structure_to_cif(
    ...     atom_array_stack,
    ...     "frame5.cif",
    ...     frame_indices=5
    ... )
    PosixPath('frame5.cif')
    >>>
    >>> save_structure_to_cif(
    ...     atom_array_stack,
    ...     "multimodel.cif",
    ...     frame_indices=[0, 10, 20, 30]
    ... )
    PosixPath('multimodel.cif')

    Notes
    -----
    - NaN coordinates will cause errors in downstream processing (e.g., CellList
      creation), so handle_nan=True is recommended.
    - When saving multimodel structures, all frames must have the same number of
      atoms and identical annotation arrays (chain_id, res_id, etc.).
    - Biotite's set_structure() automatically preserves all annotations present
      on the atom array. The extra_fields parameter is provided for API
      compatibility but annotations are saved by default.

    See Also
    --------
    biotite.structure.io.pdbx.set_structure: Underlying save function
    biotite.structure.stack: Create AtomArrayStack from multiple frames
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    structure_to_save: AtomArray | AtomArrayStack

    if frame_indices is not None:
        if not isinstance(atom_array, AtomArrayStack):
            raise ValueError("frame_indices can only be used with AtomArrayStack input")

        if isinstance(frame_indices, int):
            if frame_indices >= len(atom_array) or frame_indices < -len(atom_array):
                raise ValueError(
                    f"frame_indices {frame_indices} out of bounds for AtomArrayStack "
                    f"with {len(atom_array)} frames"
                )
            structure_to_save = cast(AtomArray, atom_array[frame_indices])
        else:
            for idx in frame_indices:
                if idx >= len(atom_array) or idx < -len(atom_array):
                    raise ValueError(
                        f"frame_indices contains {idx} which is out of bounds for "
                        f"AtomArrayStack with {len(atom_array)} frames"
                    )
            frames = [cast(AtomArray, atom_array[i]) for i in frame_indices]
            structure_to_save = stack(frames)
    else:
        structure_to_save = atom_array

    if handle_nan:
        if isinstance(structure_to_save, AtomArrayStack):
            coords: Any = structure_to_save.coord
            nan_mask = np.isnan(coords).any(axis=-1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                warnings.warn(
                    f"Found {n_nan} atom positions with NaN coordinates "
                    "across all frames. Filtering them out before saving.",
                    UserWarning,
                    stacklevel=2,
                )
                clean_mask = ~nan_mask.any(axis=0)
                structure_to_save = cast(AtomArrayStack, structure_to_save[:, clean_mask])
        else:
            coords: Any = structure_to_save.coord
            nan_mask = np.isnan(coords).any(axis=-1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                warnings.warn(
                    f"Found {n_nan} atoms with NaN coordinates. Filtering them out before saving.",
                    UserWarning,
                    stacklevel=2,
                )
                structure_to_save = cast(AtomArray, structure_to_save[~nan_mask])

    cif_file = CIFFile()
    set_structure(cif_file, structure_to_save)
    cif_file.write(str(output_path))

    return output_path


def find_all_altloc_ids(
    atom_array: AtomArray | AtomArrayStack, altloc_label: str = "altloc_id"
) -> set[str]:
    """
    Find all unique alternate location indicator (altloc) IDs in an AtomArray or AtomArrayStack.
    """
    if hasattr(atom_array, altloc_label):
        altloc_ids = np.unique(getattr(atom_array, altloc_label))
        altloc_values = getattr(atom_array, altloc_label)
        if altloc_values is None:
            raise AttributeError(
                f"atom_array.{altloc_label} exists but is None; cannot find altloc IDs"
            )
    else:
        raise AttributeError(
            "atom_array must have altloc annotation, defaults to 'altloc_id'; "
            f"you provided {altloc_label}"
        )

    return set(altloc_ids.tolist()) - BLANK_ALTLOC_IDS


def detect_altlocs(atom_array: AtomArray) -> AltlocInfo:
    """Detect alternate conformations in a structure.

    Identifies all non-default altloc IDs and creates boolean masks for each.
    Default values ("", ".", " ") are excluded from the detected altlocs.

    Parameters
    ----------
    atom_array
        The input AtomArray containing altloc_id annotations.

    Returns
    -------
    AltlocInfo
        Dataclass containing:
        - altloc_ids: Sorted list of detected altloc IDs
        - atom_masks: Dictionary mapping each altloc ID to a boolean mask of atoms

    """
    # find_all_altloc_ids has error checking
    try:
        altloc_ids = sorted(find_all_altloc_ids(atom_array))
    except AttributeError:
        logger.warning("Structure has no altloc_id annotation; assuming no alternate locations")
        return AltlocInfo(altloc_ids=[], atom_masks={})

    altloc_arr = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.altloc_id)
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {}
    for altloc in altloc_ids:
        atom_masks[altloc] = altloc_arr == altloc

    return AltlocInfo(altloc_ids=altloc_ids, atom_masks=atom_masks)


def map_altlocs_to_stack(
    atom_array: AtomArray | AtomArrayStack,
    selection: str | None = None,
    return_full_array: bool = True,
) -> tuple[AtomArrayStack, dict[str, np.ndarray]]:
    """Map alternate location indicators (altloc) to separate structures in a new AtomArrayStack.

    Note: This will raise an error if you pass an AtomArrayStack containing multiple structures.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        AtomArray or AtomArrayStack to map altlocs from.
    selection : str | None
        Optional selection string to apply to the atom array before mapping altlocs.
        If ``return_full_array`` is also True, this selection applies *only* to the altlocs.
    return_full_array : bool
        If True, return the full AtomArrayStack with all atoms, even those with no altlocs.
        If False, only return the specific atoms with altlocs.

    Returns
    -------
    tuple[AtomArrayStack, np.ndarray, np.ndarray]
        Tuple containing:

        - AtomArrayStack with separate structures for each altloc.
        - Dictionary of extra annotations that are removed to avoid conflicts, currently
          'occupancy', 'altloc_id', and 'b_factor'. Each is returned as
          n_altloc x n_res numpy arrays
    """
    if isinstance(atom_array, AtomArrayStack):
        if len(atom_array) > 1:
            raise ValueError("Cannot map altlocs with multiple structures each containing altlocs")
        atom_array = atom_array[0]

    if not hasattr(atom_array, "altloc_id") or atom_array.altloc_id is None:
        raise ValueError("The passed AtomArray | AtomArrayStack must have attribute 'altloc_id'")

    if not hasattr(atom_array, "mask") or not hasattr(atom_array, "query"):
        raise ValueError(
            "map_altlocs_to_stack requires atom_arrays loaded with "
            "atomworks.io.utils.io_utils.load_any or a similar method that provides .mask, .query"
        )

    altloc_ids = sorted(list(find_all_altloc_ids(atom_array)))
    if selection is not None and return_full_array:
        # in this case, we select atoms with no altlocs, and atoms in the selection
        # that do have altlocs construct the query (note np.isin doesn't seem to list
        # sets, hence conversion to list.
        no_altloc_mask = np.isin(atom_array.altloc_id, list(BLANK_ALTLOC_IDS))
        altloc_mask = np.isin(atom_array.altloc_id, altloc_ids)
        selection_mask = atom_array.mask(selection)
        mask = np.logical_or(no_altloc_mask, np.logical_and(altloc_mask, selection_mask))
        atom_array = atom_array[mask]
    elif selection is not None:
        atom_array = atom_array.query(selection)

    # The available altloc ids might have changed if one or more are missing from the selection.
    altloc_ids = sorted(list(find_all_altloc_ids(atom_array)))
    if len(altloc_ids) == 0:
        raise ValueError(
            f"No altlocs found in selection '{selection}' for AtomArray with altloc_id"
        )
    altloc_list = [
        select_altloc(atom_array, altloc_id, return_full_array=return_full_array)
        for altloc_id in altloc_ids
    ]
    # ensure that each structure has the same number of atoms
    # it's critical that we've updated the altloc id list above, or this will return only
    # residues with no altlocs in case one or more altloc ids is/are missing from the selection.
    atom_arrays = filter_to_common_atoms(*altloc_list)

    # remove these annotations or we cannot stack arrays but save them as output.
    extra_annotations = {}
    for key in ("occupancy", "altloc_id", "b_factor"):
        extra_annotations[key] = np.vstack([r.get_annotation(key) for r in atom_arrays])
        [r.del_annotation(key) for r in atom_arrays]

    # filter_to_common_atoms() returns a tuple of AtomArrayStack, so we need to take the first
    # (there will only be one structure in each)
    output_atom_array_stack = stack([a[0] for a in atom_arrays])

    return output_atom_array_stack, extra_annotations


def select_altloc(
    atom_array: AtomArray | AtomArrayStack, altloc_id: str, return_full_array: bool = False
) -> AtomArray | AtomArrayStack:
    """Select atoms with a specific alternate location indicator (altloc).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array.
    altloc_id : str
        The alternate location indicator to filter by.
    return_full_array : bool, optional
        If True, return the full atom array selecting atoms with either the
        specified altloc or an empty/default altloc.

    Returns
    -------
    AtomArray | AtomArrayStack
        If ``return_full_array`` is False, atoms with the specified altloc only.
        If ``return_full_array`` is True, atoms with either the specified altloc or
        an empty/default altloc (period, space, ?, or empty string). It is possible that
        an alternate altloc may be, e.g., disordered and therefore missing from the structure.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    if not (hasattr(atom_array, "altloc_id") and hasattr(atom_array, "occupancy")):
        raise AttributeError("atom_array must have `altloc_id` and `occupancy` annotations")

    if return_full_array:
        mask = np.isin(
            atom_array.altloc_id,
            list(
                {
                    altloc_id,
                }.union(BLANK_ALTLOC_IDS)
            ),
        )
    else:
        mask = atom_array.altloc_id == altloc_id

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


def select_non_hetero(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Select atoms with the hetero flag set to False.

    The hetero flag in biotite indicates whether an atom is a heteroatom
    (typically ligands, waters, or other non-standard residues). This function
    filters to return only standard protein/nucleic acid atoms.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array or stack.

    Returns
    -------
    AtomArray | AtomArrayStack
        A new array/stack with hetero=False atoms.

    Raises
    ------
    TypeError
        If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    hetero = atom_array.hetero
    if hetero is None:
        raise AttributeError("atom_array must have 'hetero' annotation")
    mask = ~hetero

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


def keep_polymer(
    atom_array: AtomArray | AtomArrayStack, pol_type: str = "peptide"
) -> AtomArray | AtomArrayStack:
    """Keep only polymer atoms from an AtomArray or AtomArrayStack.

    Filters to return only atoms where the polymer flag is True.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array or stack.
    pol_type : str
        The type of polymer to filter for. Default is "peptide".
        Options are: "peptide", "nucleotide", or "carbohydrate".
        Abbreviations are supported: "p", "pep", "n", etc.

    Returns
    -------
    AtomArray | AtomArrayStack
        A new array/stack with only polymer atoms.

    Raises
    ------
    TypeError
        If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    # TODO: fix once this is fixed: https://github.com/biotite-dev/biotite/issues/865
    if isinstance(atom_array, AtomArrayStack):
        polymer_mask = filter_polymer(atom_array[0], pol_type=pol_type)
        return atom_array[:, polymer_mask]
    else:
        polymer_mask = filter_polymer(atom_array, pol_type=pol_type)
        return atom_array[polymer_mask]


def keep_amino_acids(
    atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
    """Keep only amino acid atoms from an AtomArray or AtomArrayStack.

    Filters to return only atoms that are part of standard amino acids.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array or stack.

    Returns
    -------
    AtomArray | AtomArrayStack
        A new array/stack with only amino acid atoms.

    Raises
    ------
    TypeError
        If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    amino_acid_mask = filter_amino_acids(atom_array)

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, amino_acid_mask]
    else:
        return atom_array[amino_acid_mask]


def remove_hydrogens(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Remove hydrogen atoms from an AtomArray or AtomArrayStack.

    Filters out all atoms where the element annotation is "H" (hydrogen) or "D" (deuterium).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array or stack.

    Returns
    -------
    AtomArray | AtomArrayStack
        A new array/stack with hydrogen atoms removed.

    Raises
    ------
    TypeError
        If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    element = atom_array.element
    if element is None:
        raise AttributeError("atom_array must have 'element' annotation")
    mask = (element != "H") & (element != "D")

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


@overload
def remove_atoms_with_any_nan_coords(atom_array: AtomArrayStack) -> AtomArrayStack: ...


@overload
def remove_atoms_with_any_nan_coords(atom_array: AtomArray) -> AtomArray: ...


def remove_atoms_with_any_nan_coords(
    atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
    """
    Remove atoms with any NaN coordinates in any structure from an AtomArray or AtomArrayStack.
    """
    if not atom_array or not atom_array.shape[-1]:
        raise ValueError("Cannot remove atoms from empty AtomArray|Stack")

    if isinstance(atom_array, AtomArrayStack):
        # Partially flatten the coords so that we remove any atom from all structures if any
        # one of them has a NaN at that position.
        flattened_coords = einx.rearrange("a b c -> b (a c)", atom_array.coord)
    else:
        flattened_coords = atom_array.coord

    return atom_array[..., np.isfinite(flattened_coords).all(axis=1)]


def select_backbone(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Select only backbone atoms from an AtomArray or AtomArrayStack.

    Returns atoms with atom_name in ["C", "CA", "N", "O"], which are the
    standard protein backbone atoms.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        The input atom array or stack.

    Returns
    -------
    AtomArray | AtomArrayStack
        A new array/stack containing only backbone atoms.

    Raises
    ------
    TypeError
        If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    backbone_atoms = np.array(BACKBONE_ATOM_TYPES)
    atom_name = atom_array.atom_name
    if atom_name is None:
        raise AttributeError("atom_array must have 'atom_name' annotation")
    mask = np.isin(atom_name, backbone_atoms)

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


def make_atom_id(arr: AtomArray | AtomArrayStack) -> np.ndarray:
    """Create a unique identifier for each atom."""
    chain_id = cast(np.ndarray, arr.chain_id)
    res_id = cast(np.ndarray, arr.res_id)
    atom_name = cast(np.ndarray, arr.atom_name)
    return np.array(
        [f"{chain}_{res}_{atom}" for chain, res, atom in zip(chain_id, res_id, atom_name)]
    )


def make_normalized_atom_id(arr: AtomArray | AtomArrayStack) -> np.ndarray:
    """Like ``make_atom_id`` but with sequential 0-based residue numbering per chain.

    Handles numbering differences between representations (Boltz 0-based,
    PDB author numbering, etc.).

    Returns
    -------
    np.ndarray
        Array of ``"chainidx_seqpos_atomname"`` strings.
    """
    chain_id = cast(np.ndarray, arr.chain_id)
    res_id = cast(np.ndarray, arr.res_id)
    atom_name = cast(np.ndarray, arr.atom_name)

    # Sorted so the index assignment is stable regardless of input ordering.
    unique_chains = sorted(set(chain_id))
    chain_to_idx = {c: i for i, c in enumerate(unique_chains)}
    chain_res_to_seq: dict[tuple[str, int], int] = {}
    for chain in unique_chains:
        chain_mask = chain_id == chain
        chain_res_ids = res_id[chain_mask]
        _, first_idx = np.unique(chain_res_ids, return_index=True)
        ordered_unique = chain_res_ids[np.sort(first_idx)]
        for seq_pos, rid in enumerate(ordered_unique):
            chain_res_to_seq[(chain, int(rid))] = seq_pos

    return np.array(
        [
            f"{chain_to_idx[c]}_{chain_res_to_seq[(c, int(r))]}_{a}"
            for c, r, a in zip(chain_id, res_id, atom_name)
        ]
    )


@overload
def filter_to_common_atoms(
    *arrays: AtomArray | AtomArrayStack,
    normalize_ids: bool = ...,
    return_indices: Literal[False] = ...,
) -> tuple[AtomArrayStack, ...]: ...


@overload
def filter_to_common_atoms(
    *arrays: AtomArray | AtomArrayStack,
    normalize_ids: bool = ...,
    return_indices: Literal[True],
) -> tuple[tuple[AtomArrayStack, ...], tuple[np.ndarray, ...]]: ...


def filter_to_common_atoms(
    *arrays: AtomArray | AtomArrayStack,
    normalize_ids: bool = False,
    return_indices: bool = False,
) -> tuple[AtomArrayStack, ...] | tuple[tuple[AtomArrayStack, ...], tuple[np.ndarray, ...]]:
    """Filter multiple AtomArrays/AtomArrayStacks to only include common atoms.

    Creates unique identifiers for each atom based on chain_id, res_id, and
    atom_name, then filters all structures to include only atoms present in all.
    The returned arrays are sorted to ensure atoms are in matching order.

    Parameters
    ----------
    *arrays
        Two or more atom arrays or stacks to filter.
    normalize_ids
        If True, use sequential per-chain residue numbering instead of raw
        ``res_id``.  This handles numbering differences between representations
        (e.g. Boltz 0-based vs PDB numbering).
    return_indices
        If True, also return integer index arrays mapping each filtered
        (sorted) position back to the original array position.

    Returns
    -------
    tuple[AtomArrayStack, ...]
        Filtered versions of all input arrays containing only common atoms
        in matching order The tuple length matches the number of inputs.
        (when ``return_indices`` is False).
    tuple[tuple[AtomArrayStack, ...], tuple[np.ndarray, ...]]
        ``(filtered_arrays, index_arrays)`` when ``return_indices`` is True.
        Each ``index_array[k]`` contains the integer indices into the
        *k*-th original array such that
        ``original[k][index_array[k]]`` gives the common atoms in sorted
        order.

    Raises
    ------
    TypeError
        If inputs are not AtomArray or AtomArrayStack.
    ValueError
        If fewer than two arrays are provided.
    RuntimeError
        If no common atoms are found across all structures.

    Examples
    --------
    >>> filtered1, filtered2 = filter_to_common_atoms(array1, array2)
    >>> (f1, f2), (idx1, idx2) = filter_to_common_atoms(
    ...     model_atoms, struct_atoms, normalize_ids=True, return_indices=True
    ... )
    """
    if len(arrays) < 2:
        raise ValueError(f"At least two arrays must be provided, got {len(arrays)}")

    # Validate all inputs are correct types
    for i, array in enumerate(arrays):
        if not isinstance(array, (AtomArray, AtomArrayStack)):
            raise TypeError(
                f"Array at position {i} must be AtomArray or AtomArrayStack, got {type(array)}"
            )

    # Find common atom IDs across all structures
    id_fn = make_normalized_atom_id if normalize_ids else make_atom_id
    all_ids = [id_fn(array) for array in arrays]

    common_ids = all_ids[0]
    for ids in all_ids[1:]:
        common_ids = np.intersect1d(common_ids, ids)

    if len(common_ids) == 0:
        raise RuntimeError(f"No common atoms found across all {len(arrays)} structures")

    # Filter and sort each array
    filtered_arrays: list[AtomArrayStack] = []
    index_arrays: list[np.ndarray] = []

    for array, ids in zip(arrays, all_ids):
        # Create mask for common atoms
        mask = np.isin(ids, common_ids)
        original_indices = np.where(mask)[0]

        array = ensure_atom_array_stack(array)

        # Filter array
        filtered_array = array[:, mask]

        # Sort by atom ID to ensure matching order across arrays
        filtered_ids = ids[mask]
        sort_idx = np.argsort(filtered_ids)
        filtered_array = cast(AtomArrayStack, filtered_array[:, sort_idx])

        filtered_arrays.append(filtered_array)
        index_arrays.append(original_indices[sort_idx])

    result_arrays = tuple(filtered_arrays)
    if return_indices:
        return result_arrays, tuple(index_arrays)
    return result_arrays


def build_pairwise_altloc_arrays(
    atom_array: AtomArray, altloc_ids: list[str]
) -> dict[tuple[str, str], tuple[AtomArrayStack, AtomArrayStack]]:
    """Return ``{(id_i, id_j): (array_i, array_j)}`` pre-filtered to common atoms.

    For each unordered altloc pair we build the two per-altloc AtomArrays
    via ``select_altloc(return_full_array=True)``, which includes blank-altloc
    atoms as shared context and then run ``filter_to_common_atoms`` so the two
    inputs have identical atom order and count.

    We build per-pair rather than using ``map_altlocs_to_stack`` so residues whose
    altloc set is a subset of those in the whole structure (e.g. 2YL0 res 60–64
    carry only altlocs A and B, not C) still get scored for the pairs where they
    exist. A stack level ``filter_to_common_atoms`` would drop them entirely.

    Parameters
    ----------
    atom_array
        Input structure with ``altloc_id`` annotations.
    altloc_ids
        Altloc identifiers to enumerate over (unordered ``i < j`` pairs).

    Returns
    -------
    dict[tuple[str, str], tuple[AtomArrayStack, AtomArrayStack]]
        Mapping ``(id_i, id_j) -> (array_i, array_j)`` filtered to common atoms.
        Pairs whose atoms cannot be matched are omitted (a warning is logged).

    TODO: this helper hits the broader issue in how we
    handle structures with >2 altlocs.
    Fixing that upstream would let us replace this helper
    with a direct ``map_altlocs_to_stack`` call and remove a source of
    duplication.
    """
    pairs: dict[tuple[str, str], tuple[AtomArrayStack, AtomArrayStack]] = {}
    for i in range(len(altloc_ids)):
        for j in range(i + 1, len(altloc_ids)):
            a_i = select_altloc(atom_array, altloc_ids[i], return_full_array=True)
            a_j = select_altloc(atom_array, altloc_ids[j], return_full_array=True)
            try:
                f_i, f_j = filter_to_common_atoms(a_i, a_j)
            except RuntimeError as e:
                logger.warning(
                    f"could not match atoms between altlocs "
                    f"{altloc_ids[i]} and {altloc_ids[j]}: {e}"
                )
                continue
            pairs[(altloc_ids[i], altloc_ids[j])] = (f_i, f_j)
    return pairs
