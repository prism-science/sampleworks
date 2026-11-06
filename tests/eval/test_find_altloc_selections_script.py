"""
Integration tests for ``scripts/eval/find_altloc_selections.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "find_altloc_selections.py"
)


@pytest.fixture
def find_altloc_script():
    """Import the script module by path so tests don't require it to be
    installed on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("find_altloc_selections_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _altloc_input_csv(tmp_path: Path, resources_dir: Path, *, n_rows: int = 1) -> Path:
    """Write an ``n_rows``-row input CSV pointing at the 1vme altloc CIF.

    Skips the test if the CIF resource is missing.
    """
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    map_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    if not cif_path.exists():
        pytest.skip(f"Test resource not found: {cif_path}")
    rows = [
        {
            "name": f"1VME_{i}",
            "structure": str(cif_path),
            "density": str(map_path),
            "resolution": 1.8,
        }
        for i in range(n_rows)
    ]
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    return input_csv


def _make_args(
    *,
    input_csv: Path,
    output_file: Path,
    min_span: int = 5,
    altloc_label: str = "label_alt_id",
) -> argparse.Namespace:
    return argparse.Namespace(
        input_csv=input_csv,
        output_file=output_file,
        min_span=min_span,
        altloc_label=altloc_label,
    )


_EXPECTED_COLS = {
    "protein",
    "selection",
    "structure_pattern",
    "map_pattern",
    "base_map_dir",
    "resolution",
}


@pytest.mark.slow
def test_main_writes_expected_output_columns_and_derived_paths(
    tmp_path: Path, resources_dir: Path, find_altloc_script
):
    """Two-row happy path: verifies output schema, derived path columns, and
    that only span selections appear in the ``selection`` column."""
    input_csv = _altloc_input_csv(tmp_path, resources_dir, n_rows=2)
    output_file = tmp_path / "output.csv"

    find_altloc_script.main(_make_args(input_csv=input_csv, output_file=output_file))

    assert output_file.exists()
    df = pd.read_csv(output_file)

    assert set(df.columns) == _EXPECTED_COLS
    assert len(df) == 2

    for i, row in df.iterrows():
        assert row["protein"] == f"1VME_{i}"
        assert row["structure_pattern"] == "1vme_final_carved_edited_0.5occA_0.5occB.cif"
        assert row["map_pattern"] == "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
        assert row["base_map_dir"] == "1vme"
        assert row["resolution"] == pytest.approx(1.8)

        entries = row["selection"].split(";")
        assert any(s.startswith("chain ") and " and resi " in s for s in entries), (
            f"expected at least one old-style selection in {entries}"
        )
        assert not any("chain_id == " in s for s in entries), (
            f"the union belongs in the sibling file, not in {entries}"
        )


@pytest.mark.slow
def test_large_min_span_yields_empty_selection(
    tmp_path: Path, resources_dir: Path, find_altloc_script
):
    """An impossibly large ``min_span`` drops every span. The row is still written with
    derived columns populated and an empty ``selection`` cell."""
    input_csv = _altloc_input_csv(tmp_path, resources_dir, n_rows=1)
    output_file = tmp_path / "output.csv"

    find_altloc_script.main(
        _make_args(input_csv=input_csv, output_file=output_file, min_span=10**6)
    )

    df = pd.read_csv(output_file)
    assert len(df) == 1
    row = df.iloc[0]

    assert pd.isna(row["selection"]) or row["selection"] == ""
    assert row["protein"] == "1VME_0"
    assert row["structure_pattern"] == "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    assert row["map_pattern"] == "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    assert row["base_map_dir"] == "1vme"
