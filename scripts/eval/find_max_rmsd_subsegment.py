"""Find the maximum RMSD subsegment within each altloc selection.

This script consumes the output of ``scripts/eval/find_altloc_selections.py``
and, for each contiguous altloc span longer than ``--window-size`` residues,
identifies the contiguous subsegment of that size with the highest
RMSD between any pair of alternate conformations.

Residues are scored on the atoms shared across altlocs (via the per-pair common-atom
filtering in ``build_pairwise_altloc_arrays``), so modified residues such as CYS/CSO are
included rather than skipped. Selections may use either ``chain X and resi a-b`` or
atomworks-style syntax.

The primary output CSV preserves the setup expected by
``rscc_grid_search_script.py`` (one row per protein, semicolon joined
selections).  An optional diagnostic CSV provides per selection detail.

Usage: find_altloc_selections.py -> this script -> rscc_grid_search_script.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArrayStack, rmsd as biotite_rmsd
from joblib import delayed, Parallel
from loguru import logger
from sampleworks.eval.grid_search_eval_utils import resolve_cif_path
from sampleworks.utils.atom_array_utils import (
    build_pairwise_altloc_arrays,
    detect_altlocs,
    load_structure_with_altlocs,
)
from sampleworks.utils.structure_utils import selection_to_residues


def _find_max_rmsd_window(
    pair_arrays: dict[tuple[str, str], tuple[AtomArrayStack, AtomArrayStack]],
    chain: str,
    residues: list[int],
    window_size: int = 3,
) -> tuple[list[int], float, str] | None:
    """Slide a window over residues and find the subsegment with maximum all atom RMSD.

    Parameters
    ----------
    pair_arrays
        Output of ``build_pairwise_altloc_arrays``.
    chain
        Chain ID to filter on.
    residues
        Sorted list of actual residue IDs in the span.
    window_size
        Number of consecutive residues per window.

    Returns
    -------
    tuple or None
        ``(best_window_residues, best_rmsd, best_pair_str)`` or ``None``
        if no valid RMSD could be computed for any window.
    """
    max_rmsd = -np.inf
    max_window: list[int] | None = None
    max_pair = ""

    for w in range(len(residues) - window_size + 1):
        window_res = residues[w : w + window_size]

        for (alt_i, alt_j), (stack_i, stack_j) in pair_arrays.items():
            arr_i = stack_i[0]
            arr_j = stack_j[0]

            # arr_i and arr_j come from build_pairwise_altloc_arrays, which already ran
            # filter_to_common_atoms on the pair, so the same mask selects matched atoms in
            # both. Scoring on these shared atoms includes modified residues (e.g. CYS/CSO)
            # rather than skipping them.
            mask = (arr_i.chain_id == chain) & np.isin(arr_i.res_id, window_res)
            if mask.sum() == 0:
                continue

            rmsd_val = float(biotite_rmsd(arr_i[mask], arr_j[mask]))
            if np.isfinite(rmsd_val) and rmsd_val > max_rmsd:
                max_rmsd = rmsd_val
                max_window = window_res
                max_pair = f"{alt_i}-{alt_j}"

    if max_window is None:
        return None
    return max_window, float(max_rmsd), max_pair


def _process_structure(
    row: pd.Series,
    cif_root: Path | None,
    window_size: int = 3,
) -> list[dict]:
    """Load a structure and narrow all its selections to max RMSD subsegments."""
    protein = str(row["protein"])
    cif_path = resolve_cif_path(row, cif_root)
    if not cif_path.exists():
        logger.error(f"[{protein}] CIF file not found: {cif_path}")
        return []

    selection_field = row.get("selection", "")
    if not isinstance(selection_field, str) or not selection_field.strip():
        logger.warning(f"[{protein}] no selections in CSV row")
        return []

    logger.info(f"[{protein}] loading {cif_path}")
    atom_array = load_structure_with_altlocs(cif_path)
    altloc_info = detect_altlocs(atom_array)
    if len(altloc_info.altloc_ids) < 2:
        logger.warning(
            f"[{protein}] structure has <2 altloc IDs ({altloc_info.altloc_ids}); skipping"
        )
        return []

    pair_arrays = build_pairwise_altloc_arrays(atom_array, altloc_info.altloc_ids)
    if not pair_arrays:
        logger.warning(f"[{protein}] no valid altloc pairs could be built; skipping")
        return []

    output_rows: list[dict] = []
    for sel_str in [s.strip() for s in selection_field.split(";") if s.strip()]:
        # Resolve the selection (legacy "chain X and resi a-b" or atomworks-style) to its
        # residues. The window slides over one chain, so non-single-chain selections are skipped.
        covered = selection_to_residues(atom_array, sel_str)
        chains = {chain for chain, _ in covered}
        if len(chains) != 1:
            logger.warning(f"[{protein}] selection is empty or not single-chain: {sel_str}")
            continue
        chain = chains.pop()
        actual_res_ids = sorted(res_id for _, res_id in covered)

        out: dict = {
            "protein": protein,
            "structure_pattern": row.get("structure_pattern", ""),
            "map_pattern": row.get("map_pattern", ""),
            "base_map_dir": row.get("base_map_dir", ""),
            "resolution": row.get("resolution", ""),
            "original_selection": sel_str,
        }

        if len(actual_res_ids) <= window_size:
            out["selection"] = sel_str
            out["max_rmsd"] = float("nan")
            out["altloc_pair"] = ""
        else:
            result = _find_max_rmsd_window(pair_arrays, chain, actual_res_ids, window_size)
            if result is None:
                logger.warning(f"[{protein}] no valid RMSD window for {sel_str}; keeping original")
                out["selection"] = sel_str
                out["max_rmsd"] = float("nan")
                out["altloc_pair"] = ""
            else:
                max_res, max_rmsd, max_pair = result
                out["selection"] = f"chain {chain} and resi {max_res[0]}-{max_res[-1]}"
                out["max_rmsd"] = max_rmsd
                out["altloc_pair"] = max_pair

        output_rows.append(out)

    return output_rows


def main(args: argparse.Namespace) -> None:
    input_df = pd.read_csv(args.input_csv)
    required = {"protein", "selection"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_process_structure)(row=row, cif_root=args.cif_root, window_size=args.window_size)
        for _, row in input_df.iterrows()
    )
    all_rows: list[dict] = [r for rows in results for r in rows]

    detail_df = pd.DataFrame(all_rows)

    # Write diagnostic csv
    if args.diagnostic_file:
        args.diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(args.diagnostic_file, index=False)
        logger.info(f"Wrote {len(detail_df)} rows to {args.diagnostic_file}")

    if detail_df.empty:
        final_df = pd.DataFrame(
            columns=pd.Index(
                [
                    "protein",
                    "selection",
                    "structure_pattern",
                    "map_pattern",
                    "base_map_dir",
                    "resolution",
                ]
            )
        )
    else:
        final_df = (
            detail_df.groupby("protein", sort=False)
            .agg(
                selection=("selection", lambda s: ";".join(s)),
                structure_pattern=("structure_pattern", "first"),
                map_pattern=("map_pattern", "first"),
                base_map_dir=("base_map_dir", "first"),
                resolution=("resolution", "first"),
            )
            .reset_index()
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)
    logger.info(f"Wrote {len(final_df)} proteins to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "For each altloc selection spanning more than --window-size residues, "
            "find the contiguous subsegment with maximum pairwise all atom RMSD "
            "between altloc conformations. Narrows selections for downstream RSCC evaluation."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Output CSV from find_altloc_selections.py (columns: protein, selection, "
        "structure_pattern, map_pattern, base_map_dir, resolution).",
    )
    parser.add_argument(
        "--cif-root",
        type=Path,
        default=None,
        help="Root directory to resolve structure_pattern entries against.",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-file",
        type=Path,
        default=None,
        help="Optional per-selection diagnostic CSV with RMSD details.",
    )
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for per-structure processing (-1 = all cores).",
    )
    args = parser.parse_args()
    if args.window_size <= 0:
        parser.error("--window-size must be a positive integer")
    main(args)
