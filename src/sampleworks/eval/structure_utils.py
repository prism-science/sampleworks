import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, AtomArrayStack, from_template
from loguru import logger
from sampleworks.core.rewards.protocol import RewardInputs
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils import atom_array_utils
from sampleworks.utils.atom_array_utils import (
    BLANK_ALTLOC_IDS,
    closest_canonical_residue_name,
    load_structure_with_altlocs,
)
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.framework_utils import match_batch


try:
    from sampleworks.models.protenix.structure_processing import (
        add_terminal_oxt_atoms as _add_terminal_oxt_atoms,
        ensure_atom_array as _ensure_atom_array,
        filter_zero_occupancy as _filter_zero_occupancy,
    )

    _HAS_PROTENIX = True
except ImportError:
    _HAS_PROTENIX = False
    _add_terminal_oxt_atoms = None  # ty:ignore[invalid-assignment]
    _ensure_atom_array = None  # ty:ignore[invalid-assignment]
    _filter_zero_occupancy = None  # ty:ignore[invalid-assignment]


# TODO: standardize this more! This needs to have a specified output dataclass that
# we deal with, and define how we couple the data here to the structure we output at the end
# of sampling.
@dataclass(frozen=True, slots=True)
class SampleworksProcessedStructure:
    """Processed structure and associated model and forward model info for sampling."""

    structure: dict
    model_input: GenerativeModelInput
    input_coords: torch.Tensor
    atom_array: AtomArray
    ensemble_size: int
    reconciler: AtomReconciler
    model_atom_array: AtomArray | None = None

    @property
    def reward_atom_array(self) -> AtomArray:
        """Atom array the reward tensors and reward topology follow.

        ``model_atom_array`` is None when the model conditioning/features don't
        expose a separate atom array, i.e. the model operates on the same atom set
        as the input structure and the reconciler is an identity mapping.

        Returns
        -------
        AtomArray
            The model atom array when the model exposes one, else the structure's.
        """
        return self.model_atom_array if self.model_atom_array is not None else self.atom_array

    def to_reward_inputs(self, device: torch.device | str = "cpu") -> RewardInputs:
        """Build RewardInputs with model atom count when model atom arrays are available.

        Delegates to :meth:`RewardInputs.from_atom_array` so that element
        lookup, masking, and batching logic lives in one place.

        Reward tensors are generated from the model atom array whenever possible.
        Structure-derived B-factors are copied onto common atoms, and the
        reference coordinates come from ``self.input_coords`` (model-space).

        Parameters
        ----------
        device
            PyTorch device to place tensors on.

        Returns
        -------
        RewardInputs
        """
        atom_array_for_rewards = self.reward_atom_array

        reward_inputs = RewardInputs.from_atom_array(
            atom_array=atom_array_for_rewards,
            ensemble_size=self.ensemble_size,
            device=device,
        )

        updated_b_factors = reward_inputs.b_factors.clone()
        if self.model_atom_array is not None and self.atom_array.b_factor is not None:
            struct_b_factors = torch.as_tensor(
                cast(np.ndarray, self.atom_array.b_factor)[
                    self.reconciler.struct_indices.detach().cpu().numpy()
                ],
                device=reward_inputs.b_factors.device,
                dtype=reward_inputs.b_factors.dtype,
            )
            model_indices = self.reconciler.model_indices.to(device=reward_inputs.b_factors.device)
            updated_b_factors[..., model_indices] = struct_b_factors

        return replace(
            reward_inputs,
            b_factors=updated_b_factors,
            input_coords=self.input_coords.to(
                device=reward_inputs.input_coords.device,
                dtype=reward_inputs.input_coords.dtype,
            ),
        )


# TODO: this function could maybe be an atomworks transform/use those?
def process_structure_to_trajectory_input(
    structure: dict,
    coords_from_prior: torch.Tensor,
    features: GenerativeModelInput,
    ensemble_size: int,
) -> SampleworksProcessedStructure:
    """Convert a structure dict and model features into a ready-to-sample bundle.

    CURRENTLY: Resolves the ground-truth ``AtomArray`` from either the conditioning
    attached to *features* or the ``"asym_unit"`` key of *structure*,
    applies Protenix-specific preprocessing when needed (OXT atoms,
    zero-occupancy filtering), masks invalid atoms, and tiles the
    coordinates to *ensemble_size*.

    IDEALLY: This function should not need to be aware of the model, and should
    produce a cleaned and masked AtomArray and coordinates usable for any model.
    The model-specific logic should be handled in the model.

    Parameters
    ----------
    structure : dict
        Structure dictionary; must contain ``"asym_unit"`` if the model
        conditioning does not carry a ``true_atom_array``.  Mutated
        in-place: ``"asym_unit"`` is replaced with the cleaned array.
    coords_from_prior : torch.Tensor
        Prior sample used only to infer dtype and device for the output
        coordinate tensor. (in future this should be useful for more than that)
    features : GenerativeModelInput
        Featurized model input.  If ``features.conditioning`` exposes a
        ``true_atom_array`` attribute it is used as the reference array.
    ensemble_size : int
        Number of ensemble members; the reference coordinates are
        broadcast along a leading ``e`` dimension.

    Returns
    -------
    SampleworksProcessedStructure
        Frozen dataclass bundling the cleaned structure, model input,
        tiled coordinates, atom array, and ensemble size.
    """
    atom_array = None
    needs_protenix_preprocessing = False

    # TODO: still a bit jank, but it is better than before I think, and the transform related issues
    # will help when resolved
    conditioning = features.conditioning
    if hasattr(conditioning, "true_atom_array"):
        atom_array = conditioning.true_atom_array

    if atom_array is None:
        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")
        atom_array = ensure_atom_array_stack(structure["asym_unit"])[0]
        if conditioning is not None:
            cond_class = conditioning.__class__.__name__
            needs_protenix_preprocessing = cond_class == "ProtenixConditioning"

    # Add OXT atoms for Protenix to match the model's internal representation.
    # Only needed when falling back to structure's asym_unit (true_atom_array was None).
    if _HAS_PROTENIX and needs_protenix_preprocessing:
        assert _ensure_atom_array is not None
        assert _filter_zero_occupancy is not None
        assert _add_terminal_oxt_atoms is not None
        atom_array = _ensure_atom_array(atom_array)
        atom_array = _filter_zero_occupancy(atom_array)
        atom_array = _add_terminal_oxt_atoms(atom_array, structure.get("chain_info", {}))

    # Filter to valid atoms (nonzero occupancy, finite coords) in structure atom space.
    # The deposited structure may have zero-occupancy or NaN-coordinate atoms from
    # unresolved regions or altloc processing; these must be removed before
    # building the reconciler against the model's (already-clean) atom array.
    valid_atom_mask = get_valid_atom_mask(atom_array)
    atom_array = atom_array[valid_atom_mask]

    # Build reconciler from model and structure atom arrays.
    model_atom_array = (
        cast(AtomArray | None, conditioning.model_atom_array)
        if hasattr(conditioning, "model_atom_array")
        else None
    )
    if model_atom_array is not None:
        # here, the model_atom_array is the AtomArray with all the atoms that the model will operate
        # on, which may not overlap entirely with the deposited reference structure (atom_array)
        # atoms. The reconciler figures out which atoms are in common and deals with that
        reconciler = AtomReconciler.from_arrays(model_atom_array, atom_array)
        if reconciler.has_mismatch:
            logger.info(
                f"Atom count mismatch: model={reconciler.n_model}, "
                f"structure={reconciler.n_struct}, common={reconciler.n_common}"
            )
    else:
        reconciler = AtomReconciler.identity(len(atom_array))

    # Build model atom reference coordinates used for alignment and partial diffusion.
    struct_coords_np = np.ascontiguousarray(cast(np.ndarray, atom_array.coord))
    struct_coords = torch.from_numpy(struct_coords_np).to(
        dtype=coords_from_prior.dtype,
        device=coords_from_prior.device,
    )
    if model_atom_array is not None:
        model_template_np = cast(np.ndarray, model_atom_array.coord)

        # Replace non-finite coords with the common-atom centroid
        # TODO: use something like the RF3 processing where they put things on the nearest token
        # (this is in atomworks RF3 transform pipeline)
        struct_centroid = struct_coords_np.mean(axis=0)
        if not np.isfinite(model_template_np).all():
            logger.warning(
                "Model atom array contains non-finite coordinates; replacing with "
                "structure centroid for model-only template atoms."
            )
            bad = ~np.isfinite(model_template_np)
            model_template_np = model_template_np.copy()
            model_template_np[bad] = np.broadcast_to(struct_centroid, model_template_np.shape)[bad]
        model_template_np = np.ascontiguousarray(model_template_np)

        model_template = torch.from_numpy(model_template_np).to(
            dtype=coords_from_prior.dtype,
            device=coords_from_prior.device,
        )
        model_reference_coords = reconciler.struct_to_model(struct_coords, model_template)
    else:
        model_reference_coords = struct_coords

    input_coords = match_batch(model_reference_coords.unsqueeze(0), ensemble_size)

    structure["asym_unit"] = atom_array

    return SampleworksProcessedStructure(
        structure=structure,
        model_input=features,
        input_coords=input_coords,
        atom_array=atom_array,
        ensemble_size=ensemble_size,
        reconciler=reconciler,
        model_atom_array=model_atom_array,
    )


def get_valid_atom_mask(atom_array: AtomArray | AtomArrayStack | Any) -> Any:
    """Return a boolean mask selecting atoms usable for downstream computation.

    An atom is considered valid when it has positive occupancy and finite
    coordinates. Atoms with non-positive occupancy or with any NaN or infinite
    coordinate component are excluded.

    Parameters
    ----------
    atom_array : AtomArray or AtomArrayStack
        Biotite atom container exposing ``occupancy`` (shape ``(n_atoms,)``) and
        ``coord`` (shape ``(n_atoms, 3)`` for an ``AtomArray`` or
        ``(n_models, n_atoms, 3)`` for an ``AtomArrayStack``) annotations.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for valid atoms, broadcasting the occupancy and
        finite-coordinate criteria over the atom axis.
    """
    valid_atom_mask = atom_array.occupancy > 0
    # np.isfinite returns True for finite values, False for NaNs and infinities
    valid_atom_mask &= np.isfinite(atom_array.coord).all(axis=-1)
    return valid_atom_mask


def selection_to_residues(atom_array: AtomArray, selection: str) -> set[tuple[str, int]]:
    """Return the ``(chain_id, res_id)`` pairs covered by *selection*.

    Accepts both legacy ``chain X and resi a-b`` strings and atomworks-style
    selections (``==``, ``or``, ``in``, ...) by delegating to
    :func:`sampleworks.utils.atom_array_utils.apply_selection`, so callers can
    treat every selection syntax uniformly. Returns an empty set when the
    selection matches no atoms.

    TODO: make this work with ligands more robustly!
    """
    try:
        selected = atom_array_utils.apply_selection(atom_array, selection)
    except ValueError:
        return set()
    if selected.array_length() == 0:
        return set()
    return {
        (str(c), int(r)) for c, r in zip(np.asarray(selected.chain_id), np.asarray(selected.res_id))
    }


def extract_selection_coordinates(
    atom_array: AtomArray | AtomArrayStack, selection: str
) -> np.ndarray:
    """
    Extract coordinates for atoms matching a selection from an atomworks structure.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack Atomworks parsed structure
    selection : str Selection string like 'chain A and resi 326-339'

    Returns
    -------
    np.ndarray Coordinates of selected atoms, shape (n_atoms, 3)

    Raises
    ------
    RuntimeError: If no atoms match the selection or coordinates are invalid
    TypeError: If the "asym_unit" in `structure` is not an AtomArray or AtomArrayStack
    """
    # TODO: we will need to handle other kinds of selections later, like radius around a point.
    #   surely biotite has this capability?
    if isinstance(atom_array, AtomArrayStack):
        working_array = cast(AtomArray, atom_array[0])
    else:
        working_array = atom_array

    if not any(x in selection for x in atom_array_utils.ATOMWORKS_COMPARISON_OPS):
        mask = atom_array_utils.get_mask_from_old_selection_string(atom_array, selection)
    else:
        mask = working_array.mask(selection)

    selected_coords = cast(np.ndarray, working_array.coord)[mask]

    # VALIDATION
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No atoms matched selection: '{selection}'. "
            f"Total atoms in structure: {len(atom_array)}"
        )

    # TODO? if there are missing atoms, what happens to the computed density?
    #  Do we need to be careful here?
    # Filter out atoms with NaN or Inf coordinates (common in alt conf structures)
    finite_mask = np.isfinite(selected_coords).all(axis=1)
    if not finite_mask.all():
        n_invalid = (~finite_mask).sum()
        n_total = len(selected_coords)
        logger.warning(
            f"Filtered {n_invalid} atoms with NaN/Inf coordinates from "
            f"selection '{selection}' ({n_total - n_invalid} valid atoms remaining)"
        )
        selected_coords = selected_coords[finite_mask]

    # Check if we have any valid coordinates left
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No valid (finite) coordinates after filtering NaN/Inf from selection: '{selection}'"
        )

    return selected_coords


@overload
def get_asym_unit_from_structure(
    structure: dict, atom_array_index: None = None
) -> AtomArrayStack: ...


@overload
def get_asym_unit_from_structure(structure: dict, atom_array_index: int) -> AtomArray: ...


def get_asym_unit_from_structure(
    structure: dict, atom_array_index: int | None = None
) -> AtomArray | AtomArrayStack:
    """
    Extract the AtomArray from a structure dictionary, handling AtomArrayStack if present.
    optionally specify the index of the AtomArray in the stack (e.g. for NMR models).
    """
    atom_array = structure["asym_unit"]
    if atom_array_index is not None and isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array.get_array(atom_array_index)
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(f"Unexpected atom array type: {type(atom_array)}")
    return atom_array


def canonicalize_mixed_altloc_residues(
    atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
    """Canonicalize residue positions whose altlocs disagree on ``res_name``.

    Compositional heterogeneity, such as a canonical CYS with an alternate CSO at the same
    position, makes :func:`map_altlocs_to_stack` fail: the conformers disagree on
    ``res_name``/``hetero`` (so ``biotite.stack()`` rejects them) and the modified form carries
    extra atoms that are incompatible with the canonical residue (e.g. the CSO sulfinate
    oxygens). This function makes such positions stackable by

    1. renaming every residue at a mixed position to their shared canonical parent (via
       :func:`~sampleworks.utils.atom_array_utils.closest_canonical_residue_name`) and clearing
       ``hetero``, but only when *all*
       residue names at that position resolve to the *same* canonical parent, and
    2. dropping atoms that are not shared by every altloc at a canonicalized position, so each
       conformer ends with an identical atom set. The modified-only atoms are removed while both
       altlocs' coordinates are preserved (the alternate geometry is kept for the ensemble).

    Positions with a single, consistent ``res_name`` (e.g. an ordinary MSE with no
    heterogeneity) are left untouched. Positions whose altlocs do *not* share a single canonical
    parent (e.g. two different non-canonicals, or a canonical alongside a residue with no amino
    acid parent) are also left untouched rather than partially renamed: renaming only the
    residues that happen to map somewhere would still leave the position stacking-incompatible,
    so this logs a warning and defers the position to downstream handling instead.

    Notes
    -----
    Step 2 requires ``altloc_id`` (present on structures loaded via
    :func:`~sampleworks.utils.atom_array_utils.load_structure_with_altlocs`). It does not cover
    the case where atomworks parses the modified form as a *separate* residue under a shifted
    ``res_id`` rather than an altloc. That path still needs the CIF-level
    :func:`~sampleworks.utils.cif_utils.resolve_mixed_hetatm_atom_altlocs`.

    That CIF-level function is not simply reused here because it has a different tolerance for
    losing the modified conformer: it prepares a *single* structure to feed a ModelWrapper, where
    atomworks would otherwise insert a spurious extra residue, so it deliberately drops the
    modified altloc and keeps only the canonical one. This function instead prepares an evaluation
    *reference ensemble*, where both conformers are real experimental data that must be preserved
    as separate stack members. Unifying the two would need a shared "detect mixed compositional
    heterogeneity" core with the two resolution policies (drop-modified vs. keep-both-canonicalized)
    layered on top as options.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Reference structure loaded with altlocs.

    Returns
    -------
    AtomArray | AtomArrayStack
        Copy of the input with mixed-altloc modified residues canonicalized.
    """
    out = atom_array.copy()
    n_atoms = out.array_length()
    ins_codes = getattr(out, "ins_code", np.full(n_atoms, "")).tolist()
    positions = list(zip(out.chain_id.tolist(), out.res_id.tolist(), ins_codes))

    res_names_by_position: dict[tuple, set[str]] = defaultdict(set)
    for position, res_name in zip(positions, out.res_name.tolist()):
        res_names_by_position[position].add(res_name)
    mixed_positions = {pos for pos, names in res_names_by_position.items() if len(names) > 1}

    # 1. Only canonicalize positions whose altlocs all resolve to the same canonical parent. A
    #    partial rename that still leaves mismatched res_names at the position wouldn't be
    #    stackable anyway, so it's better to leave those untouched and warn.
    canonical_parent_by_position: dict[tuple, str] = {}
    for position in mixed_positions:
        parents = {closest_canonical_residue_name(name) for name in res_names_by_position[position]}
        if len(parents) == 1 and (parent := parents.pop()) is not None:
            canonical_parent_by_position[position] = parent
        else:
            logger.warning(
                f"Position {position} has altlocs that do not share a canonical parent "
                f"({sorted(res_names_by_position[position])}). Leaving it for downstream handling."
            )

    for i, position in enumerate(positions):
        parent = canonical_parent_by_position.get(position)
        if parent is not None:
            out.res_name[i] = parent
            out.hetero[i] = False

    # 2. Drop atoms not shared by every altloc at a canonicalized position (e.g. the modified
    #    form's extra atoms), so each conformer has an identical, stackable atom set.
    altloc_ids = getattr(out, "altloc_id", None)
    if altloc_ids is None or not canonical_parent_by_position:
        return out

    atom_names = out.atom_name.tolist()
    altlocs = altloc_ids.tolist()
    indices_by_position: dict[tuple, list[int]] = defaultdict(list)
    for i, position in enumerate(positions):
        if position in canonical_parent_by_position:
            indices_by_position[position].append(i)

    keep = np.ones(n_atoms, dtype=bool)
    for indices in indices_by_position.values():
        names_by_altloc: dict[str, set[str]] = defaultdict(set)
        for i in indices:
            if altlocs[i] not in BLANK_ALTLOC_IDS:
                names_by_altloc[altlocs[i]].add(atom_names[i])
        if len(names_by_altloc) < 2:
            continue  # need at least two conformers to know which atoms are shared
        shared_names = set.intersection(*names_by_altloc.values())
        for i in indices:
            if altlocs[i] not in BLANK_ALTLOC_IDS and atom_names[i] not in shared_names:
                keep[i] = False

    if isinstance(out, AtomArrayStack):
        return out[:, keep]
    return out[keep]


def get_reference_atomarraystack(
    protein_config: ProteinConfig, altloc_occupancies: dict[str, float]
) -> tuple[Path | str | None, AtomArrayStack | None]:
    """Load a reference structure for the given altloc occupancies.

    Parameters
    ----------
    protein_config : ProteinConfig
        Configuration for the protein.
    altloc_occupancies : dict[str, float]
        Mapping of altloc labels to occupancy values,
        e.g. ``{"A": 0.5, "B": 0.5}``.
    """
    ref_path = protein_config.get_reference_structure_path(altloc_occupancies)
    if ref_path is None:
        return None, None

    # A modified residue (e.g. CSO) recorded as an altloc at the same position as its canonical
    # form (e.g. CYS) makes map_altlocs_to_stack fail: the conformers disagree on res_name/hetero
    # so biotite.stack() rejects them. Canonicalize these residues to get map_altlocs_to_stack to
    # work
    ref_struct = canonicalize_mixed_altloc_residues(load_structure_with_altlocs(ref_path))
    if ref_struct.coord is None:
        raise ValueError(f"Unable to load coordinates from {ref_path} Please check file")
    if isinstance(ref_struct, AtomArray):
        ref_struct = from_template(ref_struct, ref_struct.coord[None, :, :])
    return ref_path, ref_struct


def get_reference_structure_coords(
    protein_config: ProteinConfig,
    protein_key: str,
    occ_list: Sequence[dict[str, float]] | None = None,
) -> dict[str, np.ndarray] | None:
    """
    This has a slightly odd function, which is to output an array of all possible coordinates
    of a structure, with altlocs mixed in. It returns NO information about which atom is which
    or whether there are duplicates. It's used for masking density maps.
    """
    if occ_list is None:
        occ_list = [{"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}]

    protein_ref_coords_list: dict[str, list[np.ndarray]] = {
        selection: [] for selection in protein_config.selection
    }
    for occ in occ_list:
        ref_path, ref_struct = get_reference_atomarraystack(protein_config, occ)
        if ref_path and ref_struct:  # if not None, it is already a validated Path object
            for selection in protein_config.selection:
                try:
                    # TODO: enumerate actual exceptions this can raise.
                    coords = extract_selection_coordinates(ref_struct, selection)
                    if not len(coords):
                        logger.warning(f"  No atoms in selection '{selection}' for {protein_key}")
                    elif not np.isfinite(coords).all():
                        logger.warning(
                            f"  NaN/Inf coordinates in selection '{selection}' for {protein_key}"
                        )
                    else:
                        protein_ref_coords_list[selection].append(coords)
                        logger.info(
                            f"  Loaded reference structure for {protein_key}: "
                            f"{len(coords)} atoms in selection '{selection}'"
                        )
                except Exception as _e:
                    _selection = selection if selection else "(none)"
                    logger.error(
                        f"  ERROR: Failed to load reference structure for {protein_key}: {_e}\n"
                        f"    Path: {ref_path}\n"
                        f"    Selection: {_selection}\n"
                        f"    Traceback: {traceback.format_exc()}"
                    )

    result = {
        k: np.vstack(protein_ref_coords_list[k])
        for k in protein_ref_coords_list
        if protein_ref_coords_list[k]
    }
    return result or None
