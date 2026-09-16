import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from pandas import DataFrame
from sampleworks.eval.grid_search_eval_utils import parse_eval_args, setup_evaluation_parameters


# TODO make more general: https://github.com/prism-science/sampleworks/issues/93
def main(args: argparse.Namespace) -> None:
    """
    Run tortoize on all trial CIF files and output residue/protein-level CSV stats.

    Parameters
    ----------
     args : argparse.Namespace
         Parsed CLI arguments from sampleworks.eval.grid_search_eval_utils.parse_eval_args().
    """

    # check that tortoize is installed and available, bail early if not.
    try:
        subprocess.call("tortoize", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("tortoize is not available, make sure you have installed it.") from None
    # The dropped variable is a list of ProteinConfigs, not used yet in this script
    all_trials, _ = setup_evaluation_parameters(args)

    # Now loop over trials with joblib and get back tuples of trial level metrics
    tortoize_results = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(get_stats_for_single_path)(trial.refined_cif_path) for trial in all_trials
    )
    if not tortoize_results:
        logger.error("No trials successfully processed, check that result files are available.")
        return

    all_residue_results, all_protein_results = tuple(zip(*tortoize_results, strict=True))

    output_file = args.grid_search_results_path / "tortoize_residues.csv"
    pd.concat(all_residue_results).to_csv(output_file, index=False)
    logger.info(f"Residue results saved to {output_file}")

    output_file = args.grid_search_results_path / "tortoize_protein_stats.csv"
    pd.concat(all_protein_results).to_csv(output_file, index=False)
    logger.info(f"Protein-level stats saved to {output_file}")


def flatten_residues(tortoize_json: dict[str, Any]) -> pd.DataFrame:
    """
    Flattens tortoize JSON into one dict per residue across all models.
    See test/1cbs.json for an example of the tortoize output JSON

    Output keys per residue:
      - model (e.g. "1")
      - asymID
      - compID
      - seqID
      - ss_type (taken from ramachandran["ss-type"] if present, else torsion["ss-type"])
      - ramachandran_z_score (ramachandran["z-score"])
      - torsion_z_score (torsion["z-score"])
    """
    out: list[dict[str, Any]] = []

    # This is possibly over-robust. Claude Sonnet wrote it and I checked/modified it.
    model_block = tortoize_json.get("model", {})
    for model_id, model_data in model_block.items():
        residues = (model_data or {}).get("residues", [])
        if not residues:
            logger.warning(f"No residues found in model {model_id}")
            continue
        for r in residues:
            rama = (r or {}).get("ramachandran", {})
            tors = (r or {}).get("torsion", {})
            pdb_info = (r or {}).get("pdb", {})

            ss_type: str | None = rama.get("ss-type", None)
            if ss_type is None:
                ss_type = tors.get("ss-type", None)

            out.append(
                {
                    "model": str(model_id),
                    "asymID": r.get("asymID", None),
                    "compID": r.get("compID", None),
                    "seqID": r.get("seqID", None),
                    "strandID": pdb_info.get("strandID", None),
                    "insCode": pdb_info.get("insCode", None),
                    "ss_type": ss_type,
                    "ramachandran_z_score": rama.get("z-score", None),
                    "torsion_z_score": tors.get("z-score", None),
                }
            )

    return pd.DataFrame(out)


def get_protein_level_z_scores(tortoize_json: dict[str, Any]) -> pd.DataFrame:
    """
    Extracts protein-level z-scores for torsion and Ramachandran angles from tortoize JSON output.
    See test/1cbs.json for an example of the tortoize output JSON


    Parameters
    ----------
    tortoize_json : dict[str, Any]
        Parsed JSON output from tortoize command.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, ramachandran_z_score, torsion_z_score,
        ramachandran_jackknife_sd, torsion_jackknife_sd.
    """

    out: list[dict[str, Any]] = []
    model_block = tortoize_json.get("model", {})
    for model_id, model_data in model_block.items():
        out.append(
            {
                "model": str(model_id),
                "ramachandran_z_score": model_data.get("ramachandran-z", None),
                "ramachandran_jackknife_sd": model_data.get("ramachandran-jackknife-sd", None),
                "torsion_z_score": model_data.get("torsion-z", None),
                "torsion_jackknife_sd": model_data.get("torsion-jackknife-sd", None),
            }
        )
    return pd.DataFrame(out)


def get_stats_for_single_path(path: Path) -> tuple[DataFrame, DataFrame]:
    """
    Run tortoize on a single CIF file and extract residue/protein-level stats.

    Parameters
    ----------
    path : Path
        Path to a CIF file to process.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (residue_df, protein_df). Both are empty DataFrames on failure.
    """
    logger.info(f"Processing {path}")
    try:
        output = subprocess.check_output(["tortoize", str(path)])
        result = json.loads(output)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to process {path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    residues = flatten_residues(result)
    if residues.empty:
        logger.warning(f"No residues found in {path}")
        return pd.DataFrame(), pd.DataFrame()

    residues["path"] = path

    protein_level_stats = get_protein_level_z_scores(result)
    protein_level_stats["path"] = path
    return residues, protein_level_stats


if __name__ == "__main__":
    argparse_description = "Crawl the workspace root for CIF files matching "
    argparse_description += "--target-filename and run tortoize on them."
    eval_args = parse_eval_args(description=argparse_description)
    main(eval_args)
