"""
Integration tests for ``scripts/eval/rscc_grid_search_script.py``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "rscc_grid_search_script.py"
)


@pytest.fixture
def rscc_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("rscc_grid_search_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_results(fx) -> pd.DataFrame:
    return pd.read_csv(fx.grid_search_results_path / "rscc_results.csv")


@pytest.mark.slow
def test_main_end_to_end_produces_csv(rscc_fixture_factory, rscc_script):
    """2 groups x 2 trials x 2 selections -> 8 rows, each with near-perfect RSCC.

    refined.cif is identical to the reference whose density is the base map, so a
    correct end-to-end run (parse, align, density, RSCC) must yield RSCC ~1.0.
    Also pins the result columns and that identical trials yield identical RSCC.
    """
    fx = rscc_fixture_factory(n_groups=2, trials_per_group=2)

    rscc_script.main(fx)

    df = _read_results(fx)
    assert len(df) == 2 * 2 * 2

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
        "error",
        "rscc",
        "base_map_path",
    }
    assert expected_cols.issubset(df.columns)

    assert df["error"].isna().all(), "no error rows expected on happy path"
    # refined.cif == the reference whose density is the base map, so each selection's RSCC is a
    # strong self-correlation: ~1.0 on CPU, but ~0.90 on GPU (the fixture builds the base map, which
    # may be on CPU or GPU depending on the test environment), and the two forward-model kernels
    # differ slightly — the smaller selection dips to ~0.897 on an H100). 0.89 keeps margin for
    # CPU/GPU variation.
    assert (df["rscc"] > 0.89).all(), f"expected strong self-correlation, got {df['rscc'].tolist()}"
    # Identical refined.cif across trials must yield identical RSCC per selection (catches a
    # cached base map being mutated/corrupted between trials).
    for sel, grp in df.groupby("selection"):
        assert np.allclose(grp["rscc"], grp["rscc"].iloc[0], atol=1e-6), (
            f"RSCC drifted across identical trials for {sel}: {grp['rscc'].tolist()}"
        )
    assert set(df["selection"]) == set(fx.selections)


@pytest.mark.slow
def test_trial_parse_failure_emits_error_rows(rscc_fixture_factory, rscc_script, monkeypatch):
    """
    Trial-level structure parsing failure results in one NaN-RSCC row per valid selection.
    """
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)

    real_parse_structure = rscc_script.parse_structure

    def flaky_parse_structure(path):
        if "refined" in str(path):
            raise RuntimeError("simulated trial parse failure")
        return real_parse_structure(path)

    monkeypatch.setattr(rscc_script, "parse_structure", flaky_parse_structure)

    rscc_script.main(fx)

    df = _read_results(fx)
    assert len(df) == len(fx.selections)
    assert df["rscc"].isna().all()
    assert df["error"].notna().all()
    assert set(df["selection"]) == set(fx.selections)


@pytest.mark.slow
def test_per_selection_failure_isolated(rscc_fixture_factory, rscc_script, monkeypatch):
    """One selection's RSCC raise must not abort the other selection's row."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)
    bad_selection = fx.selections[0]

    real_rscc = rscc_script.rscc

    class FlakyRSCC:
        """Raise once on the first call, then delegate to the real rscc."""

        def __init__(self, real):
            self.real = real
            self.target_is_next = True

        def __call__(self, a, b):
            if self.target_is_next:
                self.target_is_next = False
                raise RuntimeError("simulated rscc failure")
            return self.real(a, b)

    # The selection loop calls rscc once per selection, in CSV order. Fail the first call.
    monkeypatch.setattr(rscc_script, "rscc", FlakyRSCC(real_rscc))

    rscc_script.main(fx)

    df = _read_results(fx).set_index("selection")
    assert len(df) == len(fx.selections)
    assert np.isnan(df.loc[bad_selection, "rscc"])
    assert pd.notna(df.loc[bad_selection, "error"])

    good_selection = fx.selections[1]
    assert not np.isnan(df.loc[good_selection, "rscc"])
    assert pd.isna(df.loc[good_selection, "error"])


@pytest.mark.slow
def test_selections_missing_from_ref_coords_warn_and_produce_no_row(
    rscc_fixture_factory, rscc_script, caplog
):
    """A selection absent from the reference structure is skipped with a warning,
    while the valid selections still emit rows.
    """
    selections = (
        "chain A and resi 326-339",
        "chain A and resi 326-332",
        "chain Z and resi 9999-10000",  # no atoms in 1vme
    )
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=selections)

    with caplog.at_level(logging.WARNING):
        rscc_script.main(fx)

    df = _read_results(fx)
    assert len(df) == 2
    assert set(df["selection"]) == {selections[0], selections[1]}
    assert df["rscc"].notna().all()
    assert selections[2] in caplog.text, "the missing selection should be named in a warning"
