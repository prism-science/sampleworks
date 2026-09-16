import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sampleworks.eval.eval_dataclasses import Trial
from sampleworks.eval.grid_search_eval_utils import parse_eval_args, setup_evaluation_parameters


# TODO make more general: https://github.com/prism-science/sampleworks/issues/93
def main(args) -> None:
    # check that phenix is installed and available, bail early if not.
    try:
        subprocess.call("phenix.clashscore", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(
            "phenix.clashscore is not available, make sure phenix is installed "
            " and that you have activated it, e.g. `source phenix-dir/phenix_env.sh`"
        )
    # The dropped variable is a list of ProteinConfigs, not used yet in this script
    all_trials, _ = setup_evaluation_parameters(args)

    # Now loop over trials with joblib and get back tuples of trial level metrics
    clashscore_metrics = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(process_one_trial)(trial) for trial in all_trials
    )
    if not clashscore_metrics:
        logger.error("No trials successfully processed, check that result files are available.")
        return

    # apparently a list of empty dataframes evaluates to True, so check for that.
    clashscore_metrics = [df for df in clashscore_metrics if not df.empty]
    if not clashscore_metrics:
        logger.error("No trials produced output, check that result files are available.")
        return

    clashscore_df = pd.concat(clashscore_metrics, ignore_index=True)
    clashscore_df.to_csv(args.grid_search_results_path / "clashscore_metrics.csv", index=False)


def process_one_trial(trial: Trial) -> pd.DataFrame:
    # make sure there are no nan lines in the CIF file; this is an extra
    # precaution, even though our CIF writers should now avoid writing nans
    file_with_no_nans = trial.refined_cif_path.parent / "nonan.cif"
    json_output = trial.refined_cif_path.parent / "clashscore.json"
    logfile = trial.refined_cif_path.parent / "clashscore.log"
    logger.info(f"Removing nans from {trial.refined_cif_path}")

    with file_with_no_nans.open("w") as fn:
        grep_cmd = ["grep", "-viP", r"\bnan\b", str(trial.refined_cif_path)]
        retcode = subprocess.call(grep_cmd, stdout=fn)
    if retcode != 0:
        raise RuntimeError(f"grep failed with code {retcode}, the command was {' '.join(grep_cmd)}")

    # phenix needs to be installed and on path for this to work. Also, sh won't work with
    # phenix.clashscore because of that pesky period in the name.
    with logfile.open("w") as fn:
        # phenix.clashscore generates a JSON file with both per-model scores,
        # as well as per-model lists of clashes.
        retcode = subprocess.call(
            ["phenix.clashscore", str(file_with_no_nans), "--json-filename", str(json_output)],
            stderr=fn,
        )
    if retcode != 0:
        logger.error(f"phenix.clashscore failed, see {logfile} for details")
        return pd.DataFrame()
    return process_clashscore_json_output(json_output)


def process_clashscore_json_output(json_output: Path) -> pd.DataFrame:
    """
    Opens the JSON output file `json_output` and parses out the
    "summary_results", flattening it into rows which include the "model_name" field

    """
    with open(json_output) as f:
        json_data = json.load(f)

    model_name = json_data.get("model_name")
    # For now, we're only collecting model-level summary statistics, but
    # there are lists of specific clashes in each model too.
    summary_results = json_data.get("summary_results", {})

    rows = []
    for model_id, results in summary_results.items():
        row = {
            "model_name": model_name,
            "model_id": model_id,
            "clashscore": results.get("clashscore"),
            "num_clashes": results.get("num_clashes"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    argparse_description = "Crawl the workspace root for CIF files matching "
    argparse_description += "--target-filename and run phenix.clashscore on them."
    eval_args = parse_eval_args(description=argparse_description)
    main(eval_args)
