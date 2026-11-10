"""R-work / R-free / R-all of guided ensembles against their structure-factor targets.

Scores every ``refined.cif`` of a ``run_grid_search.py`` results tree, or one file, with
:func:`sampleworks.eval.r_factor_eval.score_ensemble`. Every trial is scored the same way,
whatever loss guided it: one engine (``--engine``), one column convention, one bulk-solvent
model. That is what makes density, ``diffuse_bragg`` and SF arms comparable.

A trial's MTZ is its protein's ``--mtz-column`` entry in ``--proteins-csv``: ``bragg_target``
(the lunus Fprotein MTZ, default) or ``mtzfile`` (the SFC MTZ). Its resolution is
``--resolution``, or else the resolution its own loss used.

A run guided with every reflection (the density ``_fullrefl`` maps, or SF without
``--exclude-free-reflections``) has seen its R-free set; compare such arms on R-all.

Every trial is scored, then all records are written to
``<results>/r_factor_results_<engine>.csv``, or to ``--output`` when the same tree is scored
against a second target.

Usage::

    python scripts/eval/r_factor_grid_search_script.py \\
        --grid-search-results-path <results> --proteins-csv <proteins.csv> \\
        --mtz-column bragg_target --expcolumns Fprotein SIGFprotein \\
        --free-flag-column R-free-flags --test-set-value 1

    python scripts/eval/r_factor_grid_search_script.py --refined <out>/refined.cif \\
        --mtz <target>.mtz --resolution 2.0 --engine lunus
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from loguru import logger
from sampleworks.eval.eval_dataclasses import Trial
from sampleworks.eval.grid_search_eval_utils import load_job_metadata, scan_grid_search_results
from sampleworks.eval.r_factor_eval import ENGINES, score_ensemble, StructureFactorTarget


def format_r(r: float | None) -> str:
    """Format an R factor for logging; ``None`` (no test set) prints as ``n/a``."""
    return "n/a" if r is None else f"{r:.4f}"


MTZ_COLUMNS = ("bragg_target", "mtzfile")


def mtz_by_protein(proteins_csv: Path, mtz_column: str) -> dict[str, Path]:
    """Map each protein name in a ``run_grid_search.py`` proteins CSV to its target MTZ.

    Parameters
    ----------
    proteins_csv
        The CSV the grid search was launched with.
    mtz_column
        Column holding the MTZ, one of ``MTZ_COLUMNS``.

    Returns
    -------
    dict of str to Path
        Rows with an empty ``mtz_column`` are left out.

    Raises
    ------
    ValueError
        If the CSV has no ``mtz_column``.
    """
    with open(proteins_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if mtz_column not in (reader.fieldnames or []):
            raise ValueError(f"{proteins_csv} has no {mtz_column!r} column: {reader.fieldnames}")
        return {row["name"]: Path(row[mtz_column]) for row in reader if row[mtz_column]}


def guided_loss_options(trial_dir: Path) -> dict[str, dict]:
    """A trial's loss options by loss name, from its ``job_metadata.json``.

    Parameters
    ----------
    trial_dir
        One grid-search trial directory.

    Returns
    -------
    dict of str to dict
        ``{loss name: loss_options}``; empty without metadata or a loss configuration.
    """
    metadata = load_job_metadata(trial_dir)
    loss_config = None if metadata is None else metadata.get("loss_config")
    if isinstance(loss_config, str):
        loss_config = json.loads(loss_config)
    if not isinstance(loss_config, dict):
        return {}
    # job_metadata.json records LossConfig.to_dict(): {loss name: {"loss_options": {...}, ...}}
    entries = cast(dict[str, dict], loss_config)
    return {name: entry["loss_options"] for name, entry in entries.items()}


def target_for_trial(
    trial: Trial, args: argparse.Namespace, mtz_lookup: dict[str, Path]
) -> StructureFactorTarget | None:
    """The target one trial is scored against, or ``None`` when it has no MTZ or resolution.

    Parameters
    ----------
    trial
        One grid-search trial.
    args
        Parsed arguments; supply the column convention and any fixed resolution.
    mtz_lookup
        Protein name to MTZ, from :func:`mtz_by_protein`.

    Returns
    -------
    StructureFactorTarget or None
        The target.
    """
    losses = guided_loss_options(trial.trial_dir)
    mtz = mtz_lookup.get(trial.protein_dir_name)
    loss_resolutions = [opts["resolution"] for opts in losses.values() if opts.get("resolution")]
    resolution = args.resolution or trial.resolution or next(iter(loss_resolutions), None)
    if mtz is None or resolution is None:
        return None
    return target_from_args(args, mtz, float(resolution))


def target_from_args(
    args: argparse.Namespace, mtz: Path, resolution: float
) -> StructureFactorTarget:
    """Build a target from the column-convention arguments."""
    amplitude, sigma = args.expcolumns or (None, None)
    return StructureFactorTarget(
        mtz=mtz,
        resolution=resolution,
        amplitude_column=amplitude,
        sigma_column=sigma,
        free_flag_column=args.free_flag_column,
        test_set_value=args.test_set_value,
        bulk_solvent=args.bulk_solvent,
    )


def score_with_oom_retry(
    path: Path, target: StructureFactorTarget, args: argparse.Namespace, **kwargs
) -> dict:
    """Score one ensemble, retrying at ``batch_partition=1`` on a CUDA OOM (SFC only).

    Parameters
    ----------
    path
        Ensemble to score.
    target
        Target to score against.
    args
        Parsed arguments: ``engine``, ``batch_partition`` and ``device``.
    **kwargs
        Forwarded to :func:`score_ensemble` (e.g. ``frame_reference``).

    Returns
    -------
    dict
        The record: R factors and everything that produced them.
    """
    partitions = (
        [args.batch_partition, 1]
        if args.engine == "sfc" and args.batch_partition > 1
        else [args.batch_partition]
    )
    for i, partition in enumerate(partitions):
        try:
            _, record = score_ensemble(
                path,
                target,
                engine=args.engine,
                batch_partition=partition,
                device=torch.device(args.device),
                **kwargs,
            )
            return record
        except torch.OutOfMemoryError:
            if i == len(partitions) - 1:
                raise
            logger.warning(f"OOM at batch_partition {partition}; retrying at 1")
            torch.cuda.empty_cache()
    raise AssertionError("unreachable")


def log_record(record: dict) -> None:
    """Log one record's R factors."""
    logger.info(
        f"{record['structure']}: R-work = {format_r(record['r_work'])}, "
        f"R-free = {format_r(record['r_free'])}, R-all = {format_r(record['r_all'])}"
    )


def score_grid_search(args: argparse.Namespace) -> int:
    """Score every trial under a grid-search tree and write all records to one CSV.

    Returns
    -------
    int
        1 if any trial failed to score, else 0.
    """
    trials = scan_grid_search_results(
        args.grid_search_results_path, target_depth=args.depth, target_filename=args.target_filename
    )
    mtz_lookup = mtz_by_protein(args.proteins_csv, args.mtz_column)
    failed = 0
    records = []
    for trial in trials:
        labels = {
            "protein": trial.protein,
            "model": trial.model,
            "method": trial.method,
            "scaler": trial.scaler,
            "loss_type": trial.loss_type,
            "ensemble_size": trial.ensemble_size,
            "guidance_weight": trial.guidance_weight,
            "trial_dir": str(trial.trial_dir),
        }
        target = target_for_trial(trial, args, mtz_lookup)
        if target is None:
            logger.info(f"skip {trial.trial_dir}: no MTZ or no resolution for it")
            continue
        frame_reference = trial.input_structure_path if args.align_to_input else None
        try:
            record = score_with_oom_retry(
                trial.refined_cif_path, target, args, frame_reference=frame_reference
            )
        except Exception as exc:  # one bad trial must not end the sweep
            logger.error(f"FAILED {trial.trial_dir}: {exc}")
            failed += 1
            continue
        log_record(record)
        records.append({**labels, **record})

    csv_path = args.output or (
        args.grid_search_results_path / f"r_factor_results_{args.engine}.csv"
    )
    pd.DataFrame(records).to_csv(csv_path, index=False)
    logger.info(f"{len(trials)} trials: {len(records)} scored, {failed} failed -> {csv_path}")
    return int(failed > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv
        Arguments for programmatic use; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--refined", type=Path, help="Score one ensemble file (needs --mtz)")
    source.add_argument(
        "--grid-search-results-path", type=Path, help="Score every trial of a results tree"
    )
    parser.add_argument("--engine", choices=ENGINES, default="sfc", help="|Fc| engine")
    parser.add_argument("--mtz", type=Path, help="Single file: target MTZ")
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Resolution cutoff (A); in grid mode defaults to each trial's own loss resolution",
    )
    parser.add_argument(
        "--expcolumns",
        nargs=2,
        default=None,
        metavar=("AMPLITUDE", "SIGMA"),
        help="Target amplitude and sigma columns; default: the MTZ's sole one of each",
    )
    parser.add_argument(
        "--free-flag-column",
        default=None,
        help="R-free flag column; default: the MTZ's sole integer column",
    )
    parser.add_argument(
        "--test-set-value",
        type=int,
        default=1,
        help="Flag value of the test set: 1 Phenix (the synthetic generators), 0 CCP4",
    )
    parser.add_argument(
        "--bulk-solvent",
        default="off",
        choices=["off", "combined", "per_conformer"],
        help="off scores |Fprotein|; combined is SFC only",
    )
    frame = parser.add_mutually_exclusive_group()
    frame.add_argument(
        "--frame-reference",
        type=Path,
        default=None,
        help="Single file: crystal-frame structure to superpose the ensemble onto first",
    )
    frame.add_argument(
        "--align-to-input",
        action="store_true",
        help="Grid mode: superpose each ensemble onto its trial's input structure first",
    )
    parser.add_argument(
        "--proteins-csv",
        type=Path,
        default=None,
        help="Grid mode (required): the proteins CSV the grid search was launched with",
    )
    parser.add_argument(
        "--mtz-column",
        choices=MTZ_COLUMNS,
        default="bragg_target",
        help="Grid mode: proteins-CSV column of each target, the lunus MTZ (bragg_target) "
        "or the SFC MTZ (mtzfile)",
    )
    parser.add_argument("--depth", type=int, default=4, help="Grid mode: trial depth")
    parser.add_argument("--target-filename", default="refined.cif", help="Grid mode: ensemble file")
    parser.add_argument(
        "--batch-partition",
        type=int,
        default=4,
        help="SFC: conformers per chunk; an OOM retries at 1",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path. Single file: also write the record there. "
        "Grid mode: write there instead of <results>/r_factor_results_<engine>.csv",
    )
    args = parser.parse_args(argv)
    if args.refined is not None and (args.mtz is None or args.resolution is None):
        parser.error("--refined needs --mtz and --resolution")
    if args.grid_search_results_path is not None and args.proteins_csv is None:
        parser.error("--grid-search-results-path needs --proteins-csv")
    if args.refined is not None and args.align_to_input:
        parser.error("--align-to-input is grid mode only; pass --frame-reference instead")
    if args.grid_search_results_path is not None and args.frame_reference is not None:
        parser.error("--frame-reference is single-file only; pass --align-to-input instead")
    return args


def main(argv: list[str] | None = None) -> int:
    """Score one ensemble, or a whole grid search.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args(argv)
    if args.grid_search_results_path is not None:
        return score_grid_search(args)
    record = score_with_oom_retry(
        args.refined,
        target_from_args(args, args.mtz, args.resolution),
        args,
        frame_reference=args.frame_reference,
    )
    log_record(record)
    if args.output is not None:
        pd.DataFrame([record]).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
