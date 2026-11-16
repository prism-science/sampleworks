"""
Unit tests for ``scripts/eval/classify_altloc_regions.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
from atomworks.io.utils.io_utils import load_any
from sampleworks.utils.atom_array_utils import detect_altlocs, load_structure_with_altlocs
from sampleworks.utils.cif_utils import find_altloc_selections

from tests.utils.atom_array_builders import altlocs_to_ensemble_cif

from .script_loader import load_script


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "classify_altloc_regions.py"
)


@pytest.fixture
def classify_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    return load_script(_SCRIPT_PATH)


def _classify_input_csv(tmp_path: Path, resources_dir: Path, selection: str) -> Path:
    """Write a one-row input CSV for the 1vme altloc CIF with the given selection.

    Skips the test if the CIF resource is missing.
    """
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Test resource not found: {cif_path}")
    input_csv = tmp_path / "input.csv"
    pd.DataFrame([{"protein": "1VME", "structure": str(cif_path), "selection": selection}]).to_csv(
        input_csv, index=False
    )
    return input_csv


def _make_args(*, input_csv: Path, output_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_csv=input_csv,
        cif_root=None,
        output_file=output_file,
        domain_shift_min_span=50,
        loop_score_metric="lddt",
        loop_score_threshold=None,
    )


def test_union_selection_is_not_classified(tmp_path: Path, resources_dir: Path, classify_script):
    """Every span of 1VME is classified once and the union appended after them is skipped.

    The selection is find_altloc_selections's own output for the 1VME resource with the union
    switched on, and the expected rows are its output with the union switched off. The
    classified rows must be exactly the spans.
    """
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Test resource not found: {cif_path}")
    spans = list(find_altloc_selections(cif_path, "label_alt_id", 1, False))
    entries = list(find_altloc_selections(cif_path, "label_alt_id", 1, True))
    assert len(entries) > len(spans), "the resource row should carry a union after its spans"

    input_csv = _classify_input_csv(tmp_path, resources_dir, ";".join(entries))
    output_file = tmp_path / "output.csv"

    classify_script.main(_make_args(input_csv=input_csv, output_file=output_file))

    df = pd.read_csv(output_file)
    assert df["selection"].tolist() == spans


@pytest.mark.parametrize("metric", ["lddt", "rmsd"])
def test_ensemble_classification_matches_single_model(
    tmp_path: Path, resources_dir: Path, metric: str, classify_script
) -> None:
    """A two-altloc structure rewritten as a 2-model ensemble gets the same classifications.

    Parameters
    ----------
    tmp_path
        Directory for the ensemble CIF.
    resources_dir
        Test resources root.
    metric
        Loop scoring metric.
    classify_script
        The script module.
    """
    scorer = classify_script.LOOP_SCORERS[metric]
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    ensemble_path = altlocs_to_ensemble_cif(cif_path, tmp_path / "ensemble.cif")
    # The fixture has exactly altlocs A and B, so the ensemble has two models and each side has
    # one conformer pair: A-B and model_0-model_1.
    assert detect_altlocs(load_structure_with_altlocs(cif_path)).altloc_ids == ["A", "B"]
    assert load_any(ensemble_path).stack_depth() == 2
    selection = ";".join(find_altloc_selections(cif_path, min_span=1))

    def classify(path: Path) -> list[dict]:
        """Classify every altloc selection of the single-model structure on ``path``.

        Parameters
        ----------
        path
            Structure to classify the selections on.

        Returns
        -------
        list[dict]
            One ``_process_structure`` output row per classified selection.
        """
        row = pd.Series({"protein": "1VME", "structure": str(path), "selection": selection})
        return classify_script._process_structure(
            input_row=row,
            cif_root=None,
            domain_shift_min_span=50,
            scoring_strategy=scorer,
            loop_score_threshold=scorer.default_threshold,
        )

    single_model, ensemble = classify(cif_path), classify(ensemble_path)
    assert single_model, "no selections were classified; the comparisons below would be vacuous"
    for key in ("selection", "classification", "n_backbone_altloc_residues", "n_altlocs"):
        assert [r[key] for r in ensemble] == [r[key] for r in single_model]
    assert [r["score"] for r in ensemble] == pytest.approx(
        [r["score"] for r in single_model], nan_ok=True
    )
    single_pairs = [json.loads(r["pair_scores"]) for r in single_model]
    ensemble_pairs = [json.loads(r["pair_scores"]) for r in ensemble]
    # Every loop row is scored by the one conformer pair; other rows have no pair scores
    assert set().union(*single_pairs) == {"A-B"}
    assert set().union(*ensemble_pairs) == {"model_0-model_1"}
