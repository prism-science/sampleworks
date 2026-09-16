"""
# Min RMSD analysis for grid search results.

For each (trial, selection), compute the minimum
all-atom RMSD over the ensemble to each reference altloc (A and B).

Writes ``min_altloc_rmsd_results.csv`` under ``--grid-search-results-path``,
mirroring the output layout of ``lddt_evaluation_script.py`` and
``rscc_grid_search_script.py``.
"""

import argparse
import itertools
import traceback

import numpy as np
import pandas as pd
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack
from joblib import delayed, Parallel
from loguru import logger
from sampleworks.eval.eval_dataclasses import ProteinConfig, Trial
from sampleworks.eval.grid_search_eval_utils import (
    parse_eval_args,
    setup_evaluation_parameters,
    translate_selection,
)
from sampleworks.metrics.rmsd import AllAtomRMSD
from sampleworks.utils.atom_array_utils import map_altlocs_to_stack
from sampleworks.utils.structure_utils import get_reference_atomarraystack


def compute_cross_segment_rmsds(
    ref_atom_array_stack: AtomArrayStack,
    pred_atom_array_stack: AtomArrayStack,
    selection: str,
) -> np.ndarray:
    """Compute segment RMSD matrix between reference altlocs and predicted models.

    For each pair ``(ref_altloc i, pred_model j)``, runs
    :meth:`AllAtomRMSD.compute` with ``superimpose=True`` and the given
    ``selection``. The returned value is the ``segment_rmsd`` for that pair,
    i.e. the all-atom RMSD restricted to atoms matching ``selection`` after a
    Kabsch superposition.

    Parameters
    ----------
    ref_atom_array_stack
        Reference AtomArrayStack, one frame per altloc.
    pred_atom_array_stack
        Predicted AtomArrayStack, one frame per ensemble member.
    selection
        Atomworks/pandas-style selection string (e.g.
        ``"chain_id == 'A' and res_id >= 326 and res_id <= 339"``) restricting
        which atoms contribute to the segment RMSD.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_altloc, n_member)`` of segment all-atom RMSDs.

    Notes
    -----
    The atom set used for segment RMSD is the maximal subset common to a single
    ``(ref_altloc, pred_member)`` pair. When altloc A and altloc B have
    different atom sets, the two rows of the returned matrix are computed over
    slightly different atom sets, so ``min_rmsd_to_A`` and ``min_rmsd_to_B``
    are not necessarily directly comparable, though they should be comparable to other
    of the same key.
    """
    n_ref = ref_atom_array_stack.stack_depth()
    n_pred = pred_atom_array_stack.stack_depth()
    segment_rmsd_matrix = np.full((n_ref, n_pred), np.nan)

    rmsd_metric = AllAtomRMSD(superimpose=True)

    for i in range(n_ref):
        ref_single = ref_atom_array_stack[i]
        for j in range(n_pred):
            pred_single = pred_atom_array_stack[j]
            result = rmsd_metric.compute(pred_single, ref_single, selection=selection)
            segment_rmsd_matrix[i, j] = result["segment_rmsd"][0]

    return segment_rmsd_matrix


def process_trial_with_selection(
    trial: Trial,
    protein_config: ProteinConfig,
    reference_atom_array_stack: AtomArrayStack,
    selection_string: str,
) -> dict[str, str | float | list[float]]:
    """Compute ``min_rmsd_to_A`` and ``min_rmsd_to_B`` for one (trial, selection).

    Loads the trial's ``refined.cif`` as an AtomArrayStack of ensemble members
    and computes an ``n_altloc x n_member`` segment RMSD matrix against
    ``reference_atom_array_stack``. ``min_rmsd_to_X`` is the per-altloc min
    over ensemble members.

    Parameters
    ----------
    trial
        The trial to score.
    protein_config
        Protein configuration; carried through for context only.
    reference_atom_array_stack
        Reference AtomArrayStack with one frame per altloc, ordered by sorted
        altloc ID. ``map_altlocs_to_stack`` produces this ordering.
    selection_string
        Pymol-style or atomworks-style selection string from the protein
        configs CSV.

    Returns
    -------
    dict
        Trial metadata plus ``selection``, ``min_rmsd_to_A``, ``min_rmsd_to_B``.

    Notes
    -----
    Extending to >2 altlocs (e.g. A/B/C) will require generalizing the output
    columns: like the LDDT eval script, the current schema is hard coded to the canonical A/B case
    and the row order produced by ``map_altlocs_to_stack`` (alphabetical altloc
    ID).
    """
    logger.debug(f"Evaluating selection {selection_string} for protein {protein_config.protein}")
    result: dict[str, str | float | list[float]] = trial.__dict__.copy()
    result["selection"] = selection_string

    predicted_atom_array_stack = ensure_atom_array_stack(load_any(trial.refined_cif_path))
    segment_rmsd_matrix = compute_cross_segment_rmsds(
        reference_atom_array_stack,
        predicted_atom_array_stack,
        translate_selection(selection_string),
    )

    # generalizing to >2 altlocs
    # will need a different output here
    result["min_rmsd_to_A"] = (
        float(np.nanmin(segment_rmsd_matrix[0])) if segment_rmsd_matrix.shape[0] >= 1 else np.nan
    )
    result["min_rmsd_to_B"] = (
        float(np.nanmin(segment_rmsd_matrix[1])) if segment_rmsd_matrix.shape[0] >= 2 else np.nan
    )

    logger.info(f"Successfully processed {trial.protein_dir_name} w/ selection {selection_string}")
    return result


def main(args: argparse.Namespace):
    all_trials, protein_configs = setup_evaluation_parameters(args)

    logger.info("Pre-loading reference structures for each protein for altloc extraction")
    reference_atom_arrays: dict[tuple[str, tuple], dict[str, AtomArrayStack]] = {}
    for protein_key, protein_config in protein_configs.items():
        for occ, sel in itertools.product(args.occupancies, protein_config.selection):
            altloc_occ = {"A": occ, "B": 1.0 - occ}
            occ_key = tuple(sorted((k, v) for k, v in altloc_occ.items() if abs(v) > 1e-6))
            ref_path, reference_proteins = get_reference_atomarraystack(protein_config, altloc_occ)
            if reference_proteins is None:
                logger.warning(
                    f"Could not find ref structure for {protein_key} and occupancies {altloc_occ}"
                )
                continue
            try:
                logger.info(
                    f"Loaded ref structure for {protein_key} "
                    f"and occupancies {altloc_occ}: {ref_path}"
                )
                reference_protein_stack, _ = map_altlocs_to_stack(
                    reference_proteins,
                    selection=translate_selection(sel),
                    return_full_array=True,
                )
                if (protein_key, occ_key) not in reference_atom_arrays:
                    reference_atom_arrays[(protein_key, occ_key)] = {}
                reference_atom_arrays[(protein_key, occ_key)][sel] = reference_protein_stack
            except Exception as e:
                logger.error(
                    f"Error loading ref structure for {protein_key} "
                    f"and occupancies {altloc_occ}: {e}"
                )
                logger.error(f"  Traceback: {traceback.format_exc()}")

    # Single pass through all trials matching the LDDT script's
    # filtered_trials / null_results pattern.
    filtered_trials: list[tuple[Trial, ProteinConfig, AtomArrayStack, str]] = []
    null_results: list[dict] = []
    for trial in all_trials:
        if trial.protein in protein_configs:
            protein = trial.protein
        elif trial.protein.upper() in protein_configs:
            protein = trial.protein.upper()
        elif trial.protein.lower() in protein_configs:
            protein = trial.protein.lower()
        else:
            logger.warning(f"Skipping protein with no configuration: {trial.protein}")
            continue

        protein_config = protein_configs[protein]
        if protein_config.protein != protein:
            raise ValueError(
                f"Protein name mismatch: expected {protein_config.protein}, got {protein}, make"
                f"sure you loaded your protein configs with ProteinConfig.from_csv()."
            )

        atom_array_key = (protein, trial.occ_key)
        if atom_array_key not in reference_atom_arrays:
            logger.warning(
                f"Skipping {trial.protein_dir_name}: no reference atom array stack available "
                f"for {trial.protein} and occupancies {trial.altloc_occupancies}."
            )
            for sel in protein_config.selection:
                trial_copy = trial.__dict__.copy()
                trial_copy["selection"] = sel
                trial_copy["min_rmsd_to_A"] = np.nan
                trial_copy["min_rmsd_to_B"] = np.nan
                null_results.append(trial_copy)
            continue

        protein_reference_atom_arrays = reference_atom_arrays[atom_array_key]
        for sel in protein_config.selection:
            if sel not in protein_reference_atom_arrays:
                logger.warning(
                    f"Skipping {trial.protein_dir_name}: no reference atom array stack available "
                    f"for {trial.protein} and occupancies {trial.altloc_occupancies}, "
                    f"selection '{sel}'."
                )
                trial_copy = trial.__dict__.copy()
                trial_copy["selection"] = sel
                trial_copy["min_rmsd_to_A"] = np.nan
                trial_copy["min_rmsd_to_B"] = np.nan
                null_results.append(trial_copy)
                continue

            filtered_trials.append((trial, protein_config, protein_reference_atom_arrays[sel], sel))

    logger.debug("Starting minimum-altloc-RMSD evaluation loop. This may take a while...")
    all_results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_trial_with_selection)(
            trial, protein_config, reference_atom_array_stack, selection
        )
        for trial, protein_config, reference_atom_array_stack, selection in filtered_trials
    )

    df = pd.DataFrame(null_results + all_results)
    df.to_csv(args.grid_search_results_path / "min_altloc_rmsd_results.csv", index=False)


if __name__ == "__main__":
    args = parse_eval_args("Evaluate minimum altloc RMSD on grid search results.")
    main(args)
