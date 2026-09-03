"""
Utilities for evaluating grid search results.
All eval scripts should use these methods to avoid any deviations.
"""

import argparse
import json
import re
import sys
import warnings
from importlib.resources import files
from pathlib import Path

import pandas as pd
from loguru import logger
from sampleworks.eval.constants import OCCUPANCY_LEVELS
from sampleworks.eval.eval_dataclasses import ProteinConfig, Trial, TrialList
from sampleworks.eval.occupancy_utils import extract_protein_and_occupancy
from sampleworks.utils.guidance_constants import StructurePredictor


def _metadata_value(
    metadata: dict[str, object],
    key: str,
    fallback: object,
    aliases: tuple[str, ...] = (),
) -> object:
    """Return the first non-empty metadata value, otherwise a fallback.

    Parameters
    ----------
    metadata
        Parsed job metadata.
    key
        Preferred field name.
    fallback
        Value used when the preferred field and aliases are absent or empty.
    aliases
        Older or alternate field names checked after ``key``.

    Returns
    -------
    object
        Resolved metadata value or ``fallback``.
    """
    for field_name in (key, *aliases):
        value = metadata.get(field_name)
        if value is not None and value != "":
            return value
    return fallback


def _metadata_float(
    metadata: dict[str, object],
    key: str,
    fallback: int | float | None,
    aliases: tuple[str, ...] = (),
) -> float | None:
    """Resolve a metadata field as a float, falling back when invalid."""
    value = _metadata_value(metadata, key, fallback, aliases)
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        logger.warning(f"Invalid numeric metadata {key}={value!r}; using {fallback!r}")
        return float(fallback) if fallback is not None else None


def _metadata_int(
    metadata: dict[str, object],
    key: str,
    fallback: int | None,
    aliases: tuple[str, ...] = (),
) -> int | None:
    """Resolve a metadata field as an integer, falling back when invalid."""
    value = _metadata_value(metadata, key, fallback, aliases)
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        logger.warning(f"Invalid integer metadata {key}={value!r}; using {fallback!r}")
        return fallback


def _metadata_occupancies(
    metadata: dict[str, object],
    fallback: dict[str, float],
) -> dict[str, float]:
    """Resolve explicit altloc occupancies, falling back to path metadata."""
    value = _metadata_value(metadata, "altloc_occupancies", fallback)
    if not isinstance(value, dict):
        logger.warning("altloc_occupancies metadata must be a JSON object; using path fallback")
        return fallback
    try:
        return {str(label).upper(): float(str(occupancy)) for label, occupancy in value.items()}
    except ValueError:
        logger.warning(
            "altloc_occupancies metadata contains a non-numeric value; using path fallback"
        )
        return fallback


def load_job_metadata(trial_dir: Path) -> dict[str, object] | None:
    """Load a trial's ``job_metadata.json`` when it is present and valid.

    Parameters
    ----------
    trial_dir
        Directory containing one grid-search trial.

    Returns
    -------
    dict[str, object] | None
        Parsed metadata, or ``None`` when no usable metadata file exists.
    """
    metadata_path = trial_dir / "job_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        with open(metadata_path) as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not load trial metadata from {metadata_path}: {exc}")
        return None
    if not isinstance(metadata, dict):
        logger.warning(f"Trial metadata must be a JSON object: {metadata_path}")
        return None
    return metadata


def resolve_cif_path(row: pd.Series, cif_root: Path | None) -> Path:
    """Resolve a CIF path from a row, preferring ``structure`` then ``structure_pattern``.

    Parameters
    ----------
    row : pd.Series
        Row containing a ``structure`` and/or ``structure_pattern`` field.
    cif_root : Path | None
        Root directory used to resolve relative paths.

    Returns
    -------
    Path
        The resolved CIF path.

    Raises
    ------
    ValueError
        If the row has neither ``structure`` nor ``structure_pattern``.

    Notes
    -----
    When resolving ``structure_pattern`` against ``cif_root``, this tries both
    ``{cif_root}/{pattern}`` (flat layout) and ``{cif_root}/{protein}/{pattern}``
    (per-protein subdirectory layout, as used by the initial_dataset processed dir).
    """
    if "structure" in row and isinstance(row["structure"], str) and row["structure"]:
        p = Path(row["structure"])
        if p.is_absolute() or p.exists():
            return p
        if cif_root is not None:
            return cif_root / p
        return p

    if (
        "structure_pattern" not in row
        or pd.isna(row["structure_pattern"])
        or not row["structure_pattern"]
    ):
        raise ValueError(f"Row has neither 'structure' nor 'structure_pattern': {row.to_dict()}")

    pattern = Path(row["structure_pattern"])
    if pattern.is_absolute():
        return pattern
    if cif_root is None:
        return pattern

    flat = cif_root / pattern
    if flat.exists():
        return flat

    protein = row.get("protein", "")
    if isinstance(protein, str) and protein:
        for candidate in (cif_root / protein / pattern, cif_root / protein.upper() / pattern):
            if candidate.exists():
                return candidate

    return flat  # fall back to flat so caller's existence check emits the right error


# TODO: this either (both) needs tests or (and) there needs to be a clearer "API"
#  for how the folder names are generated.
#  https://github.com/prism-science/sampleworks/issues/121
def parse_trial_dir(trial_dir: Path) -> dict[str, int | float | None]:
    """Parse trial directory name to extract parameters.

    Handles both:
    - fk_steering format: ens{N}_gw{W}_gd{D}
    - pure_guidance format: ens{N}_gw{W}
    """
    dir_name = trial_dir.name
    logger.debug(f"Parsing trial directory: {trial_dir}")

    # Extract ensemble size
    ens_match = re.search(r"ens(\d+)", dir_name)
    ensemble_size = int(ens_match.group(1)) if ens_match else None

    # Extract guidance weight
    gw_match = re.search(r"gw([\d.]+)", dir_name)
    guidance_weight = float(gw_match.group(1)) if gw_match else None

    # Extract gradient descent steps (for fk_steering)
    gd_match = re.search(r"gd(\d+)", dir_name)
    gd_steps = int(gd_match.group(1)) if gd_match else None

    return {
        "ensemble_size": ensemble_size,
        "guidance_weight": guidance_weight,
        "gd_steps": gd_steps,
    }


# TODO: this method is now more flexible about how it scans the grid search results directory,
#  but that means we should be more strict about the output "API" directory structure.
def scan_grid_search_results(
    current_directory: Path,
    current_depth: int = 0,
    target_depth: int = 4,
    target_filename: str = "refined.cif",
) -> TrialList:
    """Recursively scan the grid_search_results directory for all trial with refined.cif
    files.

    Parameters
    ----------
    current_directory : Path
        Path to the current directory being scanned.
    current_depth : int
        Current depth of the recursion, default 0.
    target_depth : int
        Depth where we expect to find trial output files.
    target_filename : str
        Name of the target file to look for, default "refined.cif"

    Returns
    -------
    TrialList
        List of trial metadata objects.
    """
    trials = TrialList()

    if not current_directory.exists():
        if current_depth == 0:
            logger.error(
                f"Grid search directory not found: {current_directory} at depth {current_depth}"
            )
        return trials

    # FIXME https://github.com/prism-science/sampleworks/issues/121
    # Check if we found a refined.cif file in the current directory
    refined_cif = current_directory / target_filename
    if current_depth == target_depth and refined_cif.exists():
        metadata = load_job_metadata(current_directory) or {}

        # Retain path parsing as a compatibility fallback for historical runs
        # that predate job_metadata.json.
        # Expected structure: .../protein_dir/model_dir/scaler_dir/trial_dir/refined.cif
        trial_dir = current_directory
        scaler_dir = trial_dir.parent
        model_dir = scaler_dir.parent
        protein_dir = model_dir.parent

        path_protein, path_altloc_occupancies = extract_protein_and_occupancy(protein_dir.name)
        method, path_model = get_method_and_model_name(model_dir.name)
        protein_value = _metadata_value(metadata, "protein", path_protein)
        protein = str(protein_value) if protein_value is not None else None
        altloc_occupancies = _metadata_occupancies(metadata, path_altloc_occupancies)
        model_value = _metadata_value(metadata, "model_name", path_model, aliases=("model",))
        model = str(model_value)
        method_value = _metadata_value(metadata, "method", method)
        method = str(method_value) if method_value is not None else None
        scaler = str(_metadata_value(metadata, "guidance_type", scaler_dir.name))

        params = parse_trial_dir(trial_dir)
        guidance_weight = _metadata_float(
            metadata,
            "guidance_weight",
            params["guidance_weight"],
            aliases=("step_size",),
        )
        gd_steps = _metadata_int(
            metadata,
            "num_gd_steps",
            int(params["gd_steps"]) if params["gd_steps"] is not None else None,
        )
        ensemble_size = _metadata_int(
            metadata,
            "ensemble_size",
            int(params["ensemble_size"]) if params["ensemble_size"] is not None else None,
        )
        input_structure_value = _metadata_value(metadata, "structure", "")
        density_value = _metadata_value(metadata, "density", "")
        resolution = _metadata_float(metadata, "resolution", None)

        # Validate parameters to satisfy ty
        if (
            protein is None
            or not altloc_occupancies
            or (model == StructurePredictor.BOLTZ_2 and method is None)
            or ensemble_size is None
            or (guidance_weight is None and gd_steps is None)
        ):
            logger.warning(f"Skipping trial in {trial_dir} due to missing metadata")
            return trials

        trials.append(
            Trial(
                protein=protein,
                altloc_occupancies=altloc_occupancies,
                model=model,
                method=method,
                scaler=scaler,
                ensemble_size=ensemble_size,
                guidance_weight=guidance_weight,
                gd_steps=gd_steps,
                trial_dir=trial_dir,
                refined_cif_path=refined_cif,
                protein_dir_name=protein_dir.name,
                input_structure_path=(
                    Path(str(input_structure_value)) if input_structure_value else None
                ),
                density_path=Path(str(density_value)) if density_value else None,
                resolution=resolution,
            )
        )

        return trials

    # Stop recursion if max depth reached, this should not happen, but it will prevent any
    # accidental infinite recursion if the directory structure changes in the future.
    if current_depth >= target_depth:
        return trials

    # Recurse into subdirectories
    for item in current_directory.iterdir():
        if item.is_dir() and not item.name.endswith(".json"):
            grid_search_trials = scan_grid_search_results(
                item, current_depth + 1, target_depth, target_filename=target_filename
            )
            trials.extend(grid_search_trials)

    return trials


def translate_selection(selection: str) -> str:
    """Convert a pymol-style selection (``chain A and resi N-M``) to
    atomworks/pandas style (``chain_id == 'A' and res_id >= N and res_id <= M``).

    Selections that already use atomworks comparison operators are returned
    unchanged. This temporary until selections are unified on the
    atomworks style upstream.
    """
    if any(x in selection for x in ("==", ">", "<", "<=", ">=", " in ")):
        # assume this is already atomworks/pandas style and ignore.
        return selection

    warnings.warn(
        "DEPRECATED: translate_selection converts from some pymol-like selection strings to "
        "AtomWorks selection strings, but is not guaranteed to be correct for all cases.",
        DeprecationWarning,
        stacklevel=2,
    )

    pattern = re.compile(r"chain ([A-Z]) and resi (\d+)-(\d+)")
    match = pattern.search(selection)
    if match is None:
        raise RuntimeError(f"Failed to match selection string {selection}")
    new_selection = f"chain_id == '{match.group(1)}' "
    new_selection += f"and res_id >= {match.group(2)} and res_id <= {match.group(3)}"
    return new_selection


def get_method_and_model_name(model_name: str) -> tuple[str | None, str]:
    if "MD" in model_name:
        method = "MD"
        model = model_name.replace("_MD", "")
    elif "X-RAY" in model_name:
        method = "X-RAY"
        model = model_name.replace("_X-RAY_DIFFRACTION", "")
    else:
        method = None
        model = model_name
    return method, model


def parse_eval_args(description: str | None = None):
    """
    Return a common set of arguments for grid search evaluation scripts,
    with a custom description, which is passed to argparse.ArgumentParser.

    All eval scripts should use this same framework
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--grid-search-results-path",
        type=Path,
        required=True,
        help="Path to the top-level grid search results directory, usu. called "
        "``grid_search_results``",
    )
    # not technically used everywhere yet, but requiring it future-proofs.
    parser.add_argument(
        "--grid-search-inputs-path",
        type=Path,
        required=True,
        help="Path to the directory containing the grid search inputs, in particular "
        "the protein configuration CSV file, maps, and reference structures.",
        default=None,
    )
    parser.add_argument(
        "--protein-configs-csv",
        type=Path,
        help="Path to the CSV file containing protein configurations, like "
        "``${HOME}/configs.csv``. Defaults to sampleworks/data/protein_configs.csv",
        default=files("sampleworks.data") / "protein_configs.csv",
    )
    parser.add_argument(
        "--occupancies",
        nargs="+",
        type=float,
        help=f"Occupancies to evaluate, defaults to {OCCUPANCY_LEVELS}",
        default=OCCUPANCY_LEVELS,
    )
    parser.add_argument(
        "--target-filename",
        default="refined.cif",
        help="Target filename for the CIF files to process, defaults to 'refined.cif'",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum directory depth to recurse when scanning for target CIF files.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel jobs to run. -1 uses all CPUs.",
        default=16,
    )
    parser.add_argument(
        "--selected-residues-only",
        action="store_true",
        help="Compute the refined structure's density from only the atoms in each "
        "selection (rather than the whole structure) before correlating, so RSCC "
        "reflects just the selected residues and not the surrounding structure.",
    )
    return parser.parse_args()


def setup_evaluation_parameters(
    args: argparse.Namespace,
) -> tuple[TrialList, dict[str, ProteinConfig]]:
    grid_search_dir = Path(args.grid_search_results_path)

    # Protein configurations: base map paths, structure selections, and resolutions
    protein_inputs_dir = args.grid_search_inputs_path
    protein_configs = ProteinConfig.from_csv(protein_inputs_dir, args.protein_configs_csv)

    logger.info(f"Grid search directory: {grid_search_dir}")
    logger.info(f"Proteins configured: {list(protein_configs.keys())}")

    # Scan for experiments (look for the target cif files)
    all_trials = scan_grid_search_results(
        grid_search_dir,
        target_depth=args.depth,
        target_filename=args.target_filename,
    )
    logger.info(f"Found {len(all_trials)} experiments with {args.target_filename} files")

    if all_trials:
        all_trials.summarize()  # Prints some summary stats, e.g. number of unique proteins
    else:
        logger.error("No experiments found in grid search directory. Exiting with status 1.")
        sys.exit(1)

    return all_trials, protein_configs
