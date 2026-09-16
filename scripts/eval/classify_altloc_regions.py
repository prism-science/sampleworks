"""Classify altloc regions.

This script consumes the output of ``scripts/eval/find_altloc_selections.py``
and classifies each contiguous altloc span into one of four bins:

1. ``side_chain_only`` : altloc atoms exist in the residue, but none of its
   backbone atoms have altlocs.
2. ``small_loop`` : a contiguous backbone-altloc span whose loop score is above the threshold.
3. ``large_loop`` : a contiguous backbone-altloc span whose loop score is below the threshold.
4. ``domain_shift`` : a single contiguous backbone-altloc span longer than
   ``--domain-shift-min-span`` residues (default 50). Classified before loop
   scoring.

Loop scores are calculated for every combination of altloc pairs when more than
two altlocs are present. The selected scoring strategy defines how per-residue
scores are reduced into a pair score, how pair scores are reduced into the final
loop score, and how that loop score is thresholded.

LDDT SCORER:
    For a given pair of altlocs, the score is the **equal-weighted arithmetic
    mean** of per residue backbone lDDT scores across the span:

        score = (1 / N_span_residues) * sum_k score_k

    Each ``score_k`` is the standard per-residue local lDDT from
    :class:`sampleworks.metrics.lddt.AllAtomLDDT`, which is the fraction of residue
    k's neighbor distances (within 15 Å) that are preserved between altlocs across
    the four lDDT thresholds (0.5, 1, 2, 4 Å).

    The canonical atom pair weighted lDDT would instead aggregate as
    ``sum_k(score_k * n_pairs_k) / sum_k(n_pairs_k)``. This script's
    equal residue mean is equivalent with that only when every span residue
    has the same neighbor count. The 0.75 default is calibrated for this specific
    calculation.

RMSD SCORER:
    For a given pair of altlocs, the score is the maximum of per residue
    RMSD scores across the span:

        score = max(score_k)

    Each ``score_k`` is the standard all-atom RMSD from
    :class:`sampleworks.metrics.rmsd.AllAtomRMSD`.

`Altloc pairing`: when > 2 altlocs are present, the scores above are
computed for every combination of altloc pairs and the span is classified by the
*worst* pair score, as defined by the selected scorer (the minimum for lDDT, the
maximum for RMSD).

Use ``find_altloc_selections.py --min-span 1`` to ensure single-residue side chain only
selections.
"""

import argparse
import json
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from sampleworks.eval.grid_search_eval_utils import resolve_cif_path
from sampleworks.metrics.lddt import AllAtomLDDT
from sampleworks.metrics.rmsd import AllAtomRMSD
from sampleworks.utils.atom_array_utils import (
    ATOMWORKS_COMPARISON_OPS,
    BACKBONE_ATOM_TYPES,
    BLANK_ALTLOC_IDS,
    build_pairwise_altloc_arrays,
    detect_altlocs,
    get_mask_from_old_selection_string,
    load_structure_with_altlocs,
    parse_selection_string,
)
from sampleworks.utils.structure_utils import selection_to_residues


_ATOMWORKS_CHAIN_RE = re.compile(r"chain_id\s*==\s*['\"]([^'\"]+)['\"]")


OUTPUT_COLUMNS = [
    "protein",
    "selection",
    "chain",
    "start_res",
    "end_res",
    "span_length",
    "classification",
    "score",
    "score_metric",
    "n_backbone_altloc_residues",
    "n_altlocs",
    "pair_scores",
]


def _max_contiguous_run(sorted_res_ids: np.ndarray | list[int]) -> int:
    """Return the length of the longest contiguous run of integers in a sorted list."""
    arr = np.asarray(sorted_res_ids, dtype=int)
    if arr.size == 0:
        return 0
    breaks = np.concatenate(([0], np.nonzero(np.diff(arr) != 1)[0] + 1, [arr.size]))
    return int(np.diff(breaks).max())


def _chain_from_selection(selection: str) -> str | None:
    """Extract the chain_id named by a selection string, or None if absent.

    Handles atomworks-style (``chain_id == 'A'``) and the legacy ``chain A``
    syntax accepted by ``parse_selection_string``.

    TODO: deprecate when we move all to atomworks style selections.
    """
    m = _ATOMWORKS_CHAIN_RE.search(selection)
    if m is not None:
        return m.group(1)
    if any(op in selection for op in ATOMWORKS_COMPARISON_OPS):
        # Atomworks style selection without a chain_id
        return None
    chain_id, _, _ = parse_selection_string(selection)
    return chain_id


@dataclass(frozen=True)
class SpanScorer:
    """Metric specific behavior for classification.

    Attributes
    ----------
    metric_name : str
        Short identifier for the metric (e.g. ``"lddt"``, ``"rmsd"``). Recorded in
        the ``score_metric`` output column and used in log messages.
    metric_callable : Callable[..., dict[str, Any]]
        Callable invoked with the ``predicted_atom_array_stack``,
        ``ground_truth_atom_array_stack``, and ``selection`` keyword arguments,
        returning the metric's result dictionary.
    residue_scores_key : str
        Key under which ``metric_callable``'s result holds the per-residue score
        mapping, itself keyed by ``f"{chain}{res_id}"``.
    pair_score_reducer : Callable[[list[float]], float]
        Reduces the per-residue scores of one altloc pair to a single pair score
        (mean for lDDT, max for RMSD).
    worse_pair_score : Callable[[list[float]], float]
        Reduces the per-pair scores of a span to the single worst score
        (min for lDDT, max for RMSD).
    is_small_loop_score : Callable[[float, float], bool]
        Given ``(score, threshold)``, returns True when the span is a
        ``small_loop`` (score above threshold for lDDT, below it for RMSD).
    default_threshold : float
        Threshold used when ``--loop-score-threshold`` is omitted.
    """

    metric_name: str
    metric_callable: Callable[..., dict[str, Any]]
    residue_scores_key: str
    pair_score_reducer: Callable[[list[float]], float]
    worse_pair_score: Callable[[list[float]], float]
    is_small_loop_score: Callable[[float, float], bool]
    default_threshold: float


def _score_pair_with_scorer(
    scorer: SpanScorer,
    gt_array: AtomArrayStack | AtomArray | None,
    pred_array: AtomArrayStack | AtomArray | None,
    chain: str,
    residues: list[int],
) -> float:
    """Calculate a scorer's pair score over a residue span.

    Parameters
    ----------
    scorer : SpanScorer
        Metric strategy supplying the compute callable and pair reducer.
    gt_array : AtomArrayStack | AtomArray | None
        Ground-truth altloc conformer of the pair.
    pred_array : AtomArrayStack | AtomArray | None
        Predicted altloc conformer of the pair.
    chain : str
        Chain ID the span belongs to.
    residues : list[int]
        Residue IDs of the span, scored over backbone atoms only.

    Returns
    -------
    float
        The reduced pair score, or ``nan`` when either array is missing, the span
        is empty, the metric raised, or per-residue scores could not be recovered.
        Callers treat ``nan`` as "this pair could not be scored" and skip it.
    """
    if gt_array is None or pred_array is None or not residues:
        return float("nan")

    res_clause = " or ".join(f"res_id == {r}" for r in residues)
    selection = f"chain_id == '{chain}' and ({res_clause}) and atom_name in ['C','CA','N','O']"
    try:
        result = scorer.metric_callable(
            predicted_atom_array_stack=pred_array,
            ground_truth_atom_array_stack=gt_array,
            selection=selection,
        )
    except Exception as e:
        logger.warning(
            f"{scorer.metric_name} computation failed for chain {chain} residues {residues}: {e}"
        )
        return float("nan")

    residue_scores = result.get(scorer.residue_scores_key, {})
    if not isinstance(residue_scores, dict):
        logger.warning(
            f"{scorer.metric_name} result did not contain a residue score dictionary "
            f"under key '{scorer.residue_scores_key}'"
        )
        return float("nan")

    keys = [f"{chain}{r}" for r in residues]
    missing = [k for k in keys if k not in residue_scores]
    if missing:
        logger.warning(
            f"{scorer.metric_name} result missing residues {missing} for chain {chain}. "
            f"This means the result was reduced only over the "
            f"{len(keys) - len(missing)} residues it returned"
        )

    flat = [float(residue_scores[k][0]) for k in keys if k in residue_scores]
    return float(scorer.pair_score_reducer(flat)) if flat else float("nan")


LOOP_SCORERS = {
    "lddt": SpanScorer(
        metric_name="lddt",
        metric_callable=AllAtomLDDT().compute,
        residue_scores_key="residue_lddt_scores",
        pair_score_reducer=np.mean,
        worse_pair_score=min,
        is_small_loop_score=operator.gt,
        default_threshold=0.75,
    ),
    "rmsd": SpanScorer(
        metric_name="rmsd",
        metric_callable=AllAtomRMSD(superimpose=False).compute,
        residue_scores_key="residue_rmsd_scores",
        pair_score_reducer=max,
        worse_pair_score=max,
        is_small_loop_score=operator.lt,
        default_threshold=1.0,
    ),
}


def _classify_selection(
    atom_array: AtomArray,
    pair_arrays: Mapping[
        tuple[str, str], tuple[AtomArray, AtomArray] | tuple[AtomArrayStack, AtomArrayStack]
    ],
    altloc_ids: list[str],
    selection_str: str,
    protein: str,
    structure_altloc_mask: np.ndarray,
    structure_backbone_mask: np.ndarray,
    domain_shift_min_span: int,
    scorer: SpanScorer,
    loop_score_threshold: float,
) -> tuple[dict, set[tuple[str, int]]] | None:
    """Classify one contiguous altloc selection into a conformational type.

    1. If the span has no backbone altlocs anywhere, it is classified as ``side_chain_only``.
    2. Else if the longest contiguous backbone altloc run exceeds
       ``domain_shift_min_span``, it is classified as ``domain_shift``.
    3. Else compute the per residue metric for every altloc pair over the backbone
       altloc residues in the span, reduce each pair to a single score with
       ``scorer.pair_score_reducer``, and take the worst of those with
       ``scorer.worse_pair_score``. ``scorer.is_small_loop_score`` compares that
       score against ``loop_score_threshold`` to classify the span as
       ``small_loop`` or ``large_loop``.

    Returns ``(row_dict, covered_altloc_residues)`` on success or ``None`` if the
    selection could not be applied.

    ``row_dict`` has the keys:
    ``protein``, ``selection``, ``chain``, ``start_res``, ``end_res``,
    ``span_length``, ``classification``, ``score``, ``score_metric``,
    ``n_backbone_altloc_residues``, ``n_altlocs``, and ``pair_scores`` (a
    JSON encoded ``{pair_label: score}`` map so the dict can be loaded
    through the CSV intact via ``json.loads``).

    ``covered_altloc_residues`` is the set of ``(chain_id, res_id)`` pairs in the
    span that carry any altloc, used for the caller's residue-coverage invariant
    check.
    """
    try:
        if not any(op in selection_str for op in ATOMWORKS_COMPARISON_OPS):
            sel_mask = get_mask_from_old_selection_string(atom_array, selection_str)
        else:
            sel_mask = atom_array.mask(selection_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"[{protein}] failed to apply selection '{selection_str}': {e}")
        return None

    if not sel_mask.any():
        logger.warning(f"[{protein}] selection matched no atoms: {selection_str}")
        return None

    sel_res_ids = np.unique(atom_array.res_id[sel_mask])
    sel_chain_ids = np.unique(atom_array.chain_id[sel_mask])

    # Chain is taken from the selection string. Fall back to the
    # mask-matched atoms when the selection has no chain clause.
    chain_from_sel = _chain_from_selection(selection_str)
    if chain_from_sel is None:
        if len(sel_chain_ids) != 1:
            logger.warning(
                f"{protein} selection '{selection_str}' did not specify a chain and "
                f"matched atoms that exist in these chains {sel_chain_ids.tolist()}, skipping"
            )
            return None
        chain = str(sel_chain_ids[0])
    else:
        if not (len(sel_chain_ids) == 1 and str(sel_chain_ids[0]) == chain_from_sel):
            logger.warning(
                f"{protein} selection '{selection_str}' has chain "
                f"'{chain_from_sel}' but mask matched atoms exist in chains "
                f"{sel_chain_ids.tolist()} skipping"
            )
            return None
        chain = chain_from_sel

    sel_altloc_mask = sel_mask & structure_altloc_mask
    covered_altloc_residues: set[tuple[str, int]] = {
        (str(c), int(r))
        for c, r in zip(atom_array.chain_id[sel_altloc_mask], atom_array.res_id[sel_altloc_mask])
    }

    backbone_altloc_mask = sel_altloc_mask & structure_backbone_mask
    backbone_altloc_res_ids = sorted(
        int(r) for r in np.unique(atom_array.res_id[backbone_altloc_mask])
    )
    n_backbone = len(backbone_altloc_res_ids)

    row = {
        "protein": protein,
        "selection": selection_str,
        "chain": chain,
        "start_res": int(sel_res_ids.min()),
        "end_res": int(sel_res_ids.max()),
        "span_length": int(len(sel_res_ids)),
        "n_backbone_altloc_residues": n_backbone,
        "n_altlocs": len(altloc_ids),
        # JSON encoded so the pair calculation can be loaded back through the CSV
        "pair_scores": json.dumps({}),
        "score_metric": scorer.metric_name,
        "score": float("nan"),
        "classification": "",
    }

    # Side chain only: no backbone altlocs anywhere in the span.
    if n_backbone == 0:
        row["classification"] = "side_chain_only"
        return row, covered_altloc_residues

    # Domain shift: contiguous backbone-altloc run exceeds threshold (default 50).
    if _max_contiguous_run(backbone_altloc_res_ids) > domain_shift_min_span:
        row["classification"] = "domain_shift"
        return row, covered_altloc_residues

    # Loop classification via pairwise strategy scores across all altloc pairs
    pair_scores: dict[str, float] = {}
    for i in range(len(altloc_ids)):
        for j in range(i + 1, len(altloc_ids)):
            pair = pair_arrays.get((altloc_ids[i], altloc_ids[j]))
            gt, pred = pair if pair is not None else (None, None)
            pair_scores[f"{altloc_ids[i]}-{altloc_ids[j]}"] = _score_pair_with_scorer(
                scorer, gt, pred, chain, backbone_altloc_res_ids
            )
    row["pair_scores"] = json.dumps(pair_scores)

    finite_vals = [v for v in pair_scores.values() if np.isfinite(v)]
    if not finite_vals:
        raise RuntimeError(
            f"[{protein}] could not compute {scorer.metric_name} for any altloc pair "
            f"in span '{selection_str}' "
            f"(backbone-altloc residues: {backbone_altloc_res_ids}). "
            "Refusing to emit an indeterminate classification."
        )

    worst = float(scorer.worse_pair_score(finite_vals))
    row["score"] = worst
    row["classification"] = (
        "small_loop" if scorer.is_small_loop_score(worst, loop_score_threshold) else "large_loop"
    )
    return row, covered_altloc_residues


def _process_structure(
    input_row: pd.Series,
    cif_root: Path | None,
    domain_shift_min_span: int,
    scoring_strategy: SpanScorer,
    loop_score_threshold: float,
) -> list[dict]:
    protein = str(input_row["protein"])
    cif_path = resolve_cif_path(input_row, cif_root)
    if not cif_path.exists():
        logger.error(f"[{protein}] CIF file not found: {cif_path}")
        return []

    selection_field = input_row.get("selection", "")
    if not isinstance(selection_field, str) or not selection_field.strip():
        logger.warning(f"[{protein}] no selections in CSV row for {cif_path}")
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

    structure_altloc_mask = ~np.isin(atom_array.altloc_id, list(BLANK_ALTLOC_IDS))
    structure_backbone_mask = np.isin(atom_array.atom_name, BACKBONE_ATOM_TYPES)

    # (chain, res_id) pairs that carry any altloc. Used to skip the redundant combined
    # selection below and to check coverage after classification
    all_altloc_res_ids: set[tuple[str, int]] = {
        (str(c), int(r))
        for c, r in zip(
            atom_array.chain_id[structure_altloc_mask],
            atom_array.res_id[structure_altloc_mask],
        )
    }

    rows: list[dict] = []
    classified_res_ids: set[tuple[str, int]] = set()
    for selection_str in [s.strip() for s in selection_field.split(";") if s.strip()]:
        # find_altloc_selections.py may append a combined selection unioning every altloc
        # span (atomworks-style "res_id == .. or .."). Skip only that redundant total union
        # to avoid double-counting. True discontinuous "or" selections are still
        # classified
        covered_res_ids = selection_to_residues(atom_array, selection_str)
        if covered_res_ids and covered_res_ids == all_altloc_res_ids:
            continue
        try:
            out = _classify_selection(
                atom_array=atom_array,
                pair_arrays=pair_arrays,
                altloc_ids=altloc_info.altloc_ids,
                selection_str=selection_str,
                protein=protein,
                structure_altloc_mask=structure_altloc_mask,
                structure_backbone_mask=structure_backbone_mask,
                domain_shift_min_span=domain_shift_min_span,
                scorer=scoring_strategy,
                loop_score_threshold=loop_score_threshold,
            )
        except RuntimeError as e:
            # An indeterminate span must not discard the rows already classified in this batch.
            logger.error(f"[{protein}] skipping selection '{selection_str}': {e}")
            continue
        if out is None:
            continue
        classified_row, covered = out
        rows.append(classified_row)
        classified_res_ids.update(covered)

    # residues across all classified spans should equal total unique
    # (chain, res_id) pairs that carry any altloc in the structure.
    if classified_res_ids != all_altloc_res_ids:
        missing = all_altloc_res_ids - classified_res_ids
        extra = classified_res_ids - all_altloc_res_ids
        logger.warning(
            f"[{protein}] residue coverage invariant not satisfied: "
            f"{len(missing)} altloc residues missing from classification, "
            f"{len(extra)} classified residues not in full altloc set. "
            "This typically means --min-span > 1 was used upstream."
        )

    return rows


def main(args: argparse.Namespace) -> None:
    """Run altloc region classification."""
    scoring_strategy = LOOP_SCORERS[args.loop_score_metric]
    loop_score_threshold = (
        args.loop_score_threshold
        if args.loop_score_threshold is not None
        else scoring_strategy.default_threshold
    )

    input_df = pd.read_csv(args.input_csv)
    required = {"protein", "selection"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    all_rows: list[dict] = []
    for _, row in input_df.iterrows():
        all_rows.extend(
            _process_structure(
                input_row=row,
                cif_root=args.cif_root,
                domain_shift_min_span=args.domain_shift_min_span,
                scoring_strategy=scoring_strategy,
                loop_score_threshold=loop_score_threshold,
            )
        )

    out_df = pd.DataFrame(all_rows, columns=pd.Index(OUTPUT_COLUMNS))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    logger.info(f"Wrote {len(out_df)} classified spans to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify altloc regions into side_chain_only / small_loop / "
            "large_loop / domain_shift bins. Consumes the CSV produced by "
            "find_altloc_selections.py (run with --min-span 1 to include "
            "side-chain-only regions)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Output CSV from find_altloc_selections.py (must contain 'protein' "
        "and 'selection'. May contain 'structure' or 'structure_pattern').",
    )
    parser.add_argument(
        "--cif-root",
        type=Path,
        default=None,
        help="Optional root directory to resolve 'structure_pattern' entries against.",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--domain-shift-min-span", type=int, default=50)
    parser.add_argument(
        "--loop-score-metric",
        choices=sorted(LOOP_SCORERS),
        default="lddt",
        help="Residue-level metric strategy used for small_loop / large_loop classification.",
    )
    parser.add_argument(
        "--loop-score-threshold",
        type=float,
        default=None,
        help=(
            "Threshold for the selected loop scoring strategy. Defaults to the scorer specific "
            "threshold when omitted."
        ),
    )
    args = parser.parse_args()
    main(args)
