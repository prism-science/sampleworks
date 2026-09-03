"""
# RSCC Analysis for Grid Search Results
# ported to a Python script by Marcus Collins marcus.collins@astera.org, from a notebook file
# provided by karson.chrispens@ucsf.edu

This script calculates the Real Space Correlation Coefficient (RSCC) between computed maps
from refined structures and reference (ground truth) maps for all trials in the grid
search results.

## Workflow:
1. Scan the `grid_search_results` directory for completed trials
2. For each trial with a `refined.cif`, compute the electron density map (trials are grouped
   by ``(protein, occupancy_key)`` and processed in parallel, configure with ``--n-jobs``)
3. Compare against the corresponding base map and calculate RSCC
4. Aggregate and visualize results by ensemble size, guidance weight, and scaler type

Depending on the GPU, --n-jobs=8-16 work well, and groups are spread round-robin across all
visible GPUs. ``rscc`` is populated on success, and a row gets ``rscc=nan`` only when an error
is caught (group setup, trial parsing, or during per-selection RSCC calculation), and the
``error`` column holds the reason. A CUDA RuntimeError mid-worker may still cascade to other
trials in that worker.
"""

import argparse
import copy
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray, AtomArrayStack

# Import local modules for density calculation
from joblib import delayed, Parallel
from loguru import logger
from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING
from sampleworks.eval.eval_dataclasses import ProteinConfig, Trial
from sampleworks.eval.grid_search_eval_utils import parse_eval_args, setup_evaluation_parameters
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import (
    get_asym_unit_from_structure,
    get_reference_structure_coords,
)
from sampleworks.utils.atom_array_utils import (
    ATOMWORKS_COMPARISON_OPS,
    filter_to_common_atoms,
    get_mask_from_old_selection_string,
    parse_structure,
    remove_atoms_with_any_nan_coords,
)
from sampleworks.utils.density_utils import (
    build_density_transformer,
    run_density_transformer,
)
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch


OccKey = tuple[tuple[str, float], ...]


def filter_to_selection(
    atom_array: AtomArray | AtomArrayStack, selection: str
) -> AtomArray | AtomArrayStack:
    """Restrict an atom array to the atoms matching ``selection``.

    Mirrors the masking used by
    :func:`sampleworks.eval.structure_utils.extract_selection_coordinates` (so the
    filtered atoms correspond to the same residues as the reference selection
    coordinates), while preserving every model of an ``AtomArrayStack``.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to filter. For a stack the same atom mask is applied to all models.
    selection : str
        Legacy (``chain A and resi 60-65``) or atomworks-style
        (``chain_id == 'A' and ...``) selection string.

    Returns
    -------
    AtomArray | AtomArrayStack
        The subset of ``atom_array`` matching ``selection``.

    Raises
    ------
    ValueError
        If the selection matches no atoms.
    """
    working = atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array
    if not any(op in selection for op in ATOMWORKS_COMPARISON_OPS):
        # get_mask_from_old_selection_string raises ValueError on an empty match.
        mask = get_mask_from_old_selection_string(atom_array, selection)
    else:
        mask = working.mask(selection)
        if not mask.any():
            raise ValueError(f"Selection '{selection}' matched no atoms")
    return atom_array[:, mask] if isinstance(atom_array, AtomArrayStack) else atom_array[mask]


def process_group(
    trials: list[Trial],
    protein: str,
    protein_config: ProteinConfig,
    group_ref_coords: dict[str, np.ndarray],
    base_map_path: Path,
    group_index: int,
    selected_residues_only: bool = False,
) -> list[dict]:
    """
    Process all trials sharing one (protein, occ_key) group.

    Loads the base map, builds the transformer, and parses the reference
    structure exactly once. Returns one row per (trial, valid selection),
    with ``rscc=nan`` and ``error`` populated on failure.

    Parameters
    ----------
    trials : list[Trial]
        The trials to process.
    protein : str
        The protein name.
    protein_config : ProteinConfig
        The protein configuration.
    group_ref_coords : dict[str, np.ndarray]
        The reference coordinates for the group.
    base_map_path : Path
        The path to the base map.
    group_index : int
        Position of this group in the dispatch order; used to round-robin the
        group onto one of the available GPUs.
    selected_residues_only : bool
        If True, compute each refined structure's density from only the atoms in
        that selection (rather than the whole structure) before extracting the
        region and correlating, so RSCC reflects just the selected residues. The
        density is then computed per selection instead of once per trial.

    Returns
    -------
    list[dict]
        A list of dictionaries populating the ``rscc`` and ``error`` fields for each trial.

    Raises
    ------
    ValueError
        If the base map cannot be loaded.
    """
    valid_selections = [s for s in protein_config.selection if s in group_ref_coords]
    rows: list[dict] = []

    # Spread groups across available GPUs round-robin (CPU fallback if none). Device count is
    # read in each worker so the parent never initializes CUDA before joblib forks.
    n_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{group_index % n_gpus}") if n_gpus else torch.device("cpu")

    # Load base map + transformer + reference once for the whole group.
    try:
        # Load base map for canonical unit cell,
        # don't overwrite the base map with selection map--we'll use the full map later too.
        base_xmap = protein_config.load_map(
            base_map_path,
            resolution=trials[0].resolution,
        )
        if base_xmap is None:
            raise ValueError(f"Failed to load base map from {base_map_path}")

        # The transformer is the differentiable forward model that turns atomic coordinates
        # into an electron-density map on this base map's grid, built once per group and reused.
        transformer, _ = build_density_transformer(base_xmap, em_mode=False, device=device)

        # Load the reference structure (used to align refined structures so the calculated
        # maps line up with the base map, for a correct RSCC calculation).
        ref_path = trials[0].input_structure_path or protein_config.get_reference_structure_path(
            trials[0].altloc_occupancies
        )
        if ref_path is None:
            raise ValueError(
                f"Could not find reference structure for occupancy {trials[0].altloc_occupancies}"
            )
        # parse_structure() returns only the first altloc.
        ref_structure = parse_structure(ref_path)
        ref_atom_array = get_asym_unit_from_structure(ref_structure)
        ref_atom_array = remove_atoms_with_any_nan_coords(ref_atom_array)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, AttributeError, TypeError) as e:
        logger.error(f"ERROR setting up group {protein}/{trials[0].altloc_occupancies}: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        for trial in trials:
            for selection in valid_selections:
                row = trial.__dict__.copy()
                row.update(
                    selection=selection,
                    error=str(e),
                    rscc=np.nan,
                    base_map_path=base_map_path,
                )
                rows.append(row)
        return rows

    extracted_base_cache: dict[str, np.ndarray] = {}

    # parse refined, align, and compute density once per trial.
    for trial in trials:
        try:
            structure = parse_structure(trial.refined_cif_path)
            atom_array = get_asym_unit_from_structure(structure)
            if not hasattr(atom_array, "coord") or atom_array.coord is None:
                raise AttributeError("AtomArray | AtomArrayStack is missing coordinates")

            if not hasattr(atom_array, "b_factor"):
                logger.warning(
                    f"No b-factor array found in {trial.refined_cif_path}, setting to 20."
                )
                atom_array.set_annotation("b_factor", np.full(atom_array.coord.shape[-2], 20.0))

            atom_array = remove_atoms_with_any_nan_coords(atom_array)
            # Find the common atoms with non-nan coords between the reference
            # and the refined structure.
            ref_common, pred_common = filter_to_common_atoms(ref_atom_array, atom_array)

            # Align the refined structure to the reference
            # using weighted_rigid_align_differentiable.
            # Convert to torch tensors with batch dimension.
            ref_coords_torch = torch.from_numpy(ref_common.coord).float()  # [1, n_atoms, 3]
            pred_coords_torch = torch.from_numpy(pred_common.coord).float()  # [1, n_atoms, 3]
            ref_coords_torch = match_batch(ref_coords_torch, pred_coords_torch.shape[0])
            if (
                len(ref_coords_torch.shape) != 3
                or ref_coords_torch.shape[1] != pred_coords_torch.shape[1]
            ):
                logger.error(
                    f"Shape error: ref_coords_torch: {ref_coords_torch.shape}, "
                    f"pred_coords_torch: {pred_coords_torch.shape}"
                )
                raise ValueError("ref_coords_torch and pred_coords_torch must have the same shape")

            # Create uniform weights and mask for all common atoms
            n_atoms = ref_coords_torch.shape[1]
            weights = torch.ones(1, n_atoms)
            mask = torch.ones(1, n_atoms)

            # Align predicted to reference and get the transform
            _, transform = weighted_rigid_align_differentiable(
                true_coords=pred_coords_torch,  # coords to align
                pred_coords=ref_coords_torch,  # target coords
                weights=weights,
                mask=mask,
                return_transforms=True,
                allow_gradients=False,
            )

            # Apply the transform to the entire refined structure (atom_array)
            atom_array_coords_torch = torch.from_numpy(atom_array.coord)
            aligned_coords_torch = apply_forward_transform(
                atom_array_coords_torch, transform, rotation_only=False
            )
            atom_array.coord = aligned_coords_torch.numpy()

            # Compute density from the whole aligned refined structure, shared across all
            # selections. When selected_residues_only is set, this is deferred to the
            # per-selection loop below so each map contains only that selection's atoms.
            if not selected_residues_only:
                computed_density = run_density_transformer(transformer, atom_array)
                # Shallow-copy the base xmap so .array can be rebound without touching the cache.
                # XMap.extract_tight reads self.array live, so the two wrappers stay independent.
                computed_xmap = copy.copy(base_xmap)
                computed_xmap.array = computed_density.cpu().numpy()
                if computed_xmap.array.shape != base_xmap.array.shape:
                    raise ValueError(
                        f"density shape {computed_xmap.array.shape} does not match base map "
                        f"shape {base_xmap.array.shape}"
                    )
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            RuntimeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.error(f"ERROR processing trial {trial.trial_dir}: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            for selection in valid_selections:
                row = trial.__dict__.copy()
                row.update(
                    selection=selection,
                    error=str(e),
                    rscc=np.nan,
                    base_map_path=base_map_path,
                )
                rows.append(row)
            continue

        # Per selection, extract base region (cache) + computed region, compute RSCC
        for selection in valid_selections:
            sel_coords = group_ref_coords[selection]
            row = trial.__dict__.copy()
            row.update(selection=selection, error=None, base_map_path=base_map_path)
            try:
                extracted_base = extracted_base_cache.get(selection)
                if extracted_base is None:
                    _, extracted_base = base_xmap.extract_tight(
                        sel_coords, padding=DEFAULT_SELECTION_PADDING
                    )
                    if extracted_base is None or extracted_base.shape[0] == 0:
                        raise ValueError(f"Extracted base map empty for selection {selection}")
                    extracted_base_cache[selection] = extracted_base

                if selected_residues_only:
                    # Compute density from only this selection's atoms on the shared grid, so
                    # the extracted region carries no signal from the surrounding structure.
                    selected_atoms = filter_to_selection(atom_array, selection)
                    selected_density = run_density_transformer(transformer, selected_atoms)
                    computed_xmap = copy.copy(base_xmap)
                    computed_xmap.array = selected_density.cpu().numpy()

                _, extracted_computed = computed_xmap.extract_tight(
                    sel_coords, padding=DEFAULT_SELECTION_PADDING
                )

                # Validate extraction
                if extracted_computed is None or extracted_computed.shape[0] == 0:
                    raise ValueError("Extracted computed map is empty")

                # Calculate RSCC on extracted regions
                row["rscc"] = rscc(extracted_base, extracted_computed)
            except Exception as e:
                logger.error(f"ERROR processing {trial.trial_dir} selection {selection}: {e}")
                row["error"] = str(e)
                row["rscc"] = np.nan  # this is the default, but better to be explicit.
            rows.append(row)

    return rows


def main(args: argparse.Namespace):
    all_trials, protein_configs = setup_evaluation_parameters(args)

    logger.info("Pre-loading reference structures for each protein for coordinate extraction")
    ref_coords: dict[tuple[str, str], np.ndarray] = {}
    for protein_key, protein_config in protein_configs.items():
        # NOTE THAT THIS will be by default include all altlocs, as we use them to create a mask
        # for where to judge the maps' correlation.
        protein_ref_coords = get_reference_structure_coords(protein_config, protein_key)
        if protein_ref_coords is not None:
            for selection in protein_ref_coords.keys():
                ref_coords[(protein_key, selection)] = protein_ref_coords[selection]

    # Calculate RSCC for all trials
    logger.info("Calculating RSCC values for all trials...")
    logger.warning(
        "Note: RSCC is computed on the region around altloc residues (defined by selection)"
    )

    # Sort so all trials sharing a (protein, occ_key) are contiguous, then build groups.
    # Resolve protein name once per group and slice ref_coords for each protein.
    groups: list[tuple[str, list[Trial], Path, dict[str, np.ndarray]]] = []
    group_index: dict[tuple[str, OccKey, Path | None], int] = {}
    for trial in sorted(all_trials, key=lambda t: (t.protein, t.occ_key)):
        if trial.protein in protein_configs:
            protein = trial.protein
        elif trial.protein.upper() in protein_configs:
            protein = trial.protein.upper()
        else:
            logger.warning(f"Skipping protein with no configuration: {trial.protein}")
            continue
        key = (protein, trial.occ_key, trial.density_path)
        idx = group_index.get(key)
        if idx is None:
            protein_config = protein_configs[protein]
            base_map_path = trial.density_path or protein_config.get_base_map_path_for_occupancy(
                trial.altloc_occupancies
            )
            if base_map_path is None:
                logger.warning(
                    f"Skipping group {protein}/{trial.altloc_occupancies}: base map not found"
                )
                group_index[key] = -1
                continue
            group_ref_coords = {
                s: ref_coords[(protein, s)]
                for s in protein_config.selection
                if (protein, s) in ref_coords
            }
            dropped_selections = [
                s for s in protein_config.selection if (protein, s) not in ref_coords
            ]
            if dropped_selections:
                logger.warning(
                    f"[{protein}] skipping {len(dropped_selections)} selection(s) with no "
                    f"reference coords: {dropped_selections}"
                )
            if not group_ref_coords:
                logger.warning(
                    f"Skipping group {protein}/{trial.altloc_occupancies}: "
                    f"no reference structure for any configured selection"
                )
                group_index[key] = -1
                continue
            group_index[key] = len(groups)
            groups.append((protein, [trial], base_map_path, group_ref_coords))
        elif idx >= 0:
            groups[idx][1].append(trial)

    group_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_group)(
            trials,
            protein,
            protein_configs[protein],
            group_ref_coords,
            base_map_path,
            i,
            selected_residues_only=args.selected_residues_only,
        )
        for i, (protein, trials, base_map_path, group_ref_coords) in enumerate(groups)
    )
    results = [row for rows in group_results for row in rows]

    logger.info(f"\nCompleted RSCC calculation for {len(results)} trials")

    # Create DataFrame from results
    df = pd.DataFrame(results)
    df.to_csv(args.grid_search_results_path / "rscc_results.csv", index=False)

    if not df.empty:
        # Remove error column for display if present
        drop_cols = [
            "trial_dir",
            "refined_cif_path",
            "base_map_path",
            "error",
            "protein_dir_name",
        ]

        logger.info("Results Summary:")
        logger.info(df.drop(drop_cols, axis=1).head(20).to_string())  # noqa

        logger.info("\n\nSummary Statistics by Protein and Scaler:")
        summary = (
            df.groupby(["protein", "scaler"])["rscc"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        logger.info(summary)


if __name__ == "__main__":
    args = parse_eval_args("Evaluate RSCC on grid search results.")
    main(args)
