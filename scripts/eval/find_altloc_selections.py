import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sampleworks.utils.cif_utils import find_altloc_selections


def _process_row(row: pd.Series, altloc_label: str, min_span: int) -> pd.Series:
    cif_file = row["structure"]
    spans, unions = find_altloc_selections(cif_file, altloc_label, min_span)
    if not spans:
        logger.warning(
            f"No spans emitted for {cif_file} (min_span={min_span}) "
            f"altlocs may exist but none met these criteria."
        )

    # The column names here are defined by the input requirements of scripts like
    # rscc_grid_search_script.py
    output = {
        "protein": row["name"],  # this should be the RSCB ID
        "selection": ";".join(spans),
        "structure_pattern": Path(cif_file).name,
        "map_pattern": Path(row["density"]).name,
        "base_map_dir": Path(row["density"]).parent.name,
        "resolution": row["resolution"],
        # Carried so main() can swap it into the sibling union file, then dropped.
        "union_selection": ";".join(unions),
    }
    return pd.Series(output)


def main(args):
    """
    Write two output files with rows consisting of
    RCSB ID, selections for altlocs, and path to the corresponding CIF file,

    The spans go to ``--output-file`` and the per-chain unions to a sibling ``_union`` file.

    This script will likely change until we settle on a final input/output
    results directory structure.
    """
    input_df = pd.read_csv(args.input_csv)
    # apply() on an empty frame returns the input's columns, not _process_row's, so the two
    # files below would be built from the wrong columns. Fail here instead, where the cause is.
    if input_df.empty:
        raise ValueError(f"No rows in {args.input_csv}; there is nothing to find selections for")

    output = input_df.apply(
        _process_row,
        altloc_label=args.altloc_label,
        min_span=args.min_span,
        axis=1,
    )
    span_df = output.drop(columns="union_selection")
    span_df.to_csv(args.output_file, index=False)

    union_file = args.output_file.with_stem(f"{args.output_file.stem}_union")
    span_df.assign(selection=output["union_selection"]).to_csv(union_file, index=False)
    logger.info(f"Wrote {len(span_df)} rows to {args.output_file} and to {union_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the input CSV config file used for grid search",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--min-span", type=int, default=5)
    parser.add_argument("--altloc-label", type=str, default="label_alt_id")
    args = parser.parse_args()
    main(args)
