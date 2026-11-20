"""
Integration tests for ``scripts/eval/rmsd_evaluation_script.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.script_loader import load_script


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "rmsd_evaluation_script.py"
)


@pytest.fixture
def rmsd_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    return load_script(_SCRIPT_PATH)


def _make_args(fx, occupancies: list[float] | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        grid_search_results_path=fx.grid_search_results_path,
        grid_search_inputs_path=fx.grid_search_inputs_path,
        protein_configs_csv=fx.protein_configs_csv,
        occupancies=occupancies if occupancies is not None else [0.5],
        target_filename="refined.cif",
        depth=4,
        n_jobs=1,
    )


def _read_results(fx) -> pd.DataFrame:
    return pd.read_csv(fx.grid_search_results_path / "min_altloc_rmsd_results.csv")


def test_main_end_to_end_produces_csv(rscc_fixture_factory, rmsd_script):
    """1 group x 1 trial x 1 selection produces one row with the expected columns.

    The fixture is the same CIF as both the reference and
    the trial's ``refined.cif``. ``load_any`` defaults pick a single altloc, so the trial is
    effectively the altloc-A reference. That is, ``min_rmsd_to_A`` must be ~0 while
    ``min_rmsd_to_B`` must be strictly positive.
    """
    fx = rscc_fixture_factory(
        n_groups=1, trials_per_group=1, selections=("chain A and resi 326-339",)
    )

    rmsd_script.main(_make_args(fx))

    df = _read_results(fx)
    expected_cols = {
        "protein",
        "altloc_occupancies",
        "model",
        "scaler",
        "ensemble_size",
        "trial_dir",
        "refined_cif_path",
        "protein_dir_name",
        "selection",
        "min_rmsd_to_A",
        "min_rmsd_to_B",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["min_rmsd_to_A"] == pytest.approx(0.0, abs=1e-4)
    assert row["min_rmsd_to_B"] > 0.1


def test_missing_reference_produces_nan_rows(rscc_fixture_factory, rmsd_script):
    """Trials whose ``(protein, occ_key)`` has no reference produce NaN rows.

    Requesting occupancies that the fixture has no reference structure for
    causes the skip on every trial. The script must still emit
    one row per (trial, selection), with ``min_rmsd_to_A`` /
    ``min_rmsd_to_B`` = NaN, rather than aborting.
    """
    fx = rscc_fixture_factory(
        n_groups=1, trials_per_group=1, selections=("chain A and resi 326-339",)
    )

    rmsd_script.main(_make_args(fx, occupancies=[0.9]))

    df = _read_results(fx)
    assert len(df) == 1
    row = df.iloc[0]
    assert np.isnan(row["min_rmsd_to_A"])
    assert np.isnan(row["min_rmsd_to_B"])
