"""
Unit tests for ``scripts/eval/classify_altloc_regions.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
from sampleworks.utils.cif_utils import find_altloc_selections

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
