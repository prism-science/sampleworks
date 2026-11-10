"""Tests for :mod:`sampleworks.eval.r_factor_eval` and its grid-search script.

The target is 1vme through the lunus generator, its altlocs expanded into a two-configuration
ensemble with a Phenix-convention R-free set: a self-consistent pair of ensemble and target.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from biotite.structure import AtomArrayStack
from biotite.structure.io import save_structure

from tests.eval.script_loader import load_script


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

from sampleworks.eval.r_factor_eval import score_ensemble, StructureFactorTarget  # noqa: E402


SOURCE = "1vme_final.cif"
RESOLUTION = 1.8
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "r_factor_grid_search_script.py"


@pytest.fixture(scope="module")
def configurations(resources_dir: Path) -> AtomArrayStack:
    """1vme as the lunus generator loads it: one configuration per altloc, waters stripped."""
    from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations
    from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ, SyntheticStructureOptions

    return load_configurations(
        resources_dir / "1vme" / SOURCE,
        BatchRowForMTZ(filename=SOURCE),
        "default",
        structure_options=SyntheticStructureOptions(strip_waters=True),
        altlocs_as_models=True,
    )


@pytest.fixture(scope="module")
def target_mtz(resources_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ensemble-mean ``Fprotein`` of :func:`configurations`, 10% flagged free (flag 1)."""
    from sampleworks.synthetic.generate_synthetic_sf_lunus import _process_single_row
    from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ, SyntheticStructureOptions

    record = _process_single_row(
        row=BatchRowForMTZ(filename=SOURCE),
        base_dir=resources_dir / "1vme",
        output_dir=tmp_path_factory.mktemp("r_factor_target"),
        resolution=RESOLUTION,
        occupancy_mode="default",
        test_fraction=0.1,
        seed=0,
        device=torch.device("cpu"),
        structure_options=SyntheticStructureOptions(strip_waters=True),
        altlocs_as_models=True,
    )
    assert record["status"] == "success", record.get("error")
    return Path(record["output_path"])


def write_ensemble(stack: AtomArrayStack, coords: np.ndarray, path: Path) -> Path:
    """Write ``stack`` with ``coords`` ``[n_models, n_atoms, 3]`` as a multi-model cif."""
    stack = stack.copy()
    stack.coord = coords.astype(np.float32)
    save_structure(path, stack)
    return path


@pytest.fixture(scope="module")
def ensemble_cif(configurations: AtomArrayStack, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The configurations the target was computed from."""
    path = tmp_path_factory.mktemp("ensemble") / "refined.cif"
    return write_ensemble(configurations, configurations.coord, path)


@pytest.fixture(scope="module")
def perturbed_cif(configurations: AtomArrayStack, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The configurations with 0.3 A Gaussian noise on every coordinate."""
    noise = np.random.default_rng(0).normal(0.0, 0.3, configurations.coord.shape)
    path = tmp_path_factory.mktemp("perturbed") / "refined.cif"
    return write_ensemble(configurations, configurations.coord + noise, path)


def target(mtz: Path, test_set_value: int = 1) -> StructureFactorTarget:
    """The convention the lunus generator writes, with the test-set flag value chosen."""
    return StructureFactorTarget(
        mtz=mtz,
        resolution=RESOLUTION,
        amplitude_column="Fprotein",
        sigma_column="SIGFprotein",
        free_flag_column="R-free-flags",
        test_set_value=test_set_value,
    )


def test_self_consistent_target_scores_zero(ensemble_cif, target_mtz):
    r, record = score_ensemble(
        ensemble_cif, target(target_mtz), engine="lunus", device=torch.device("cpu")
    )
    assert r.all < 1e-3
    assert r.work < 1e-3 and r.free is not None and r.free < 1e-3
    assert record["n_conformers"] == 2


def test_perturbed_ensemble_scores_worse(perturbed_cif, target_mtz):
    r, _ = score_ensemble(
        perturbed_cif, target(target_mtz), engine="lunus", device=torch.device("cpu")
    )
    assert r.all > 0.05


def test_free_flag_convention_changes_r_free(perturbed_cif, target_mtz):
    """Flag value 0 swaps the sets: R-free moves, R-all over the same reflections does not."""
    device = torch.device("cpu")
    phenix, _ = score_ensemble(perturbed_cif, target(target_mtz), engine="lunus", device=device)
    ccp4, _ = score_ensemble(
        perturbed_cif, target(target_mtz, test_set_value=0), engine="lunus", device=device
    )
    assert ccp4.free != phenix.free
    assert ccp4.all == pytest.approx(phenix.all, rel=1e-6)


def test_columns_resolve_from_the_mtz(ensemble_cif, target_mtz):
    """With no columns named, the MTZ's sole amplitude, sigma and flag columns are used."""
    r, record = score_ensemble(
        ensemble_cif,
        StructureFactorTarget(mtz=target_mtz, resolution=RESOLUTION),
        engine="lunus",
        device=torch.device("cpu"),
    )
    assert (record["amplitude_column"], record["free_flag_column"]) == ("Fprotein", "R-free-flags")
    assert r.all < 1e-3


def test_sfc_engine_agrees_with_lunus(ensemble_cif, target_mtz, device):
    """SFC scores the lunus target near zero too: the two engines compute the same Fprotein."""
    r, _ = score_ensemble(ensemble_cif, target(target_mtz), engine="sfc", device=device)
    assert r.all < 1e-3


def test_frame_reference_undoes_a_translation(
    configurations, ensemble_cif, target_mtz, tmp_path: Path
):
    device = torch.device("cpu")
    shifted = write_ensemble(
        configurations, configurations.coord + np.array([5.0, -3.0, 2.0]), tmp_path / "shifted.cif"
    )
    off_frame, _ = score_ensemble(shifted, target(target_mtz), engine="lunus", device=device)
    realigned, _ = score_ensemble(
        shifted, target(target_mtz), engine="lunus", frame_reference=ensemble_cif, device=device
    )
    assert off_frame.all > 0.3
    assert realigned.all < 0.05


def test_script_scores_one_file(ensemble_cif, target_mtz, tmp_path: Path):
    out = tmp_path / "r.csv"
    status = load_script(SCRIPT).main(
        [
            "--refined", str(ensemble_cif),
            "--mtz", str(target_mtz),
            "--resolution", str(RESOLUTION),
            "--engine", "lunus",
            "--device", "cpu",
            "--output", str(out),
        ]
    )  # fmt: skip
    record = pd.read_csv(out).iloc[0]
    assert status == 0
    assert record["engine"] == "lunus" and record["r_all"] < 1e-3
