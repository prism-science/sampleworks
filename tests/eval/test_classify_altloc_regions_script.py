"""
Integration tests for ``scripts/eval/classify_altloc_regions.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from .script_loader import load_script


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "classify_altloc_regions.py"
)


@pytest.fixture
def classify_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    return load_script(_SCRIPT_PATH)


def _classify_input_csv(tmp_path: Path, resources_dir: Path, selection: str) -> Path:
    """Write a one-row input CSV for the 1vme altloc CIF with the given selection cell.

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


def test_repeated_selection_is_classified_once(
    tmp_path: Path, resources_dir: Path, classify_script
):
    """A selection cell holding a span and the combined selection that repeats it gives one row.

    find_altloc_selections.py appends a per-chain combined selection after the individual
    spans. Here the cell holds one span, 1VME residue 82, and a combined selection naming the
    same residue, so the second string repeats the first and must be skipped. Residue 82's
    altlocs are on side-chain atoms only, so no loop scoring runs.
    """
    selection = "chain A and resi 82-82;chain_id == 'A' and ((res_id == 82))"
    input_csv = _classify_input_csv(tmp_path, resources_dir, selection)
    output_file = tmp_path / "output.csv"

    classify_script.main(_make_args(input_csv=input_csv, output_file=output_file))

    df = pd.read_csv(output_file)
    assert df[["selection", "classification"]].values.tolist() == [
        ["chain A and resi 82-82", "side_chain_only"]
    ]
