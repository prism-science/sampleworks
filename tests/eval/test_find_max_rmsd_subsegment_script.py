"""Tests for ``scripts/eval/find_max_rmsd_subsegment.py``."""

from pathlib import Path

import pandas as pd
import pytest
from atomworks.io.utils.io_utils import load_any
from sampleworks.utils.atom_array_utils import detect_altlocs, load_structure_with_altlocs
from sampleworks.utils.cif_utils import find_altloc_selections

from tests.eval.script_loader import load_script
from tests.utils.atom_array_builders import altlocs_to_ensemble_cif


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "find_max_rmsd_subsegment.py"
)


def test_ensemble_window_matches_single_model(tmp_path: Path, resources_dir: Path) -> None:
    """A two-altloc structure rewritten as a 2-model ensemble gets the same max-RMSD windows.

    Parameters
    ----------
    tmp_path
        Directory for the ensemble CIF.
    resources_dir
        Test resources root.
    """
    rmsd_script = load_script(_SCRIPT_PATH)
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    ensemble_path = altlocs_to_ensemble_cif(cif_path, tmp_path / "ensemble.cif")
    # The fixture has exactly altlocs A and B, so the ensemble has two models and each side has
    # one conformer pair: A-B and model_0-model_1.
    assert detect_altlocs(load_structure_with_altlocs(cif_path)).altloc_ids == ["A", "B"]
    assert load_any(ensemble_path).stack_depth() == 2
    selection = ";".join(find_altloc_selections(cif_path))

    def windows(path: Path) -> list[dict]:
        """Narrow every altloc selection of the single-model structure on ``path``.

        Parameters
        ----------
        path
            Structure to score the selections on.

        Returns
        -------
        list[dict]
            One ``_process_structure`` output row per selection.
        """
        row = pd.Series({"protein": "1VME", "structure": str(path), "selection": selection})
        return rmsd_script._process_structure(row, cif_root=None, window_size=3)

    single_model, ensemble = windows(cif_path), windows(ensemble_path)
    assert [r["selection"] for r in ensemble] == [r["selection"] for r in single_model]
    assert [r["max_rmsd"] for r in ensemble] == pytest.approx(
        [r["max_rmsd"] for r in single_model], nan_ok=True
    )
    assert {r["altloc_pair"] for r in single_model if r["altloc_pair"]} == {"A-B"}
    assert {r["altloc_pair"] for r in ensemble if r["altloc_pair"]} == {"model_0-model_1"}
