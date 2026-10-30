"""
The evaluators read the input's numbering from the author columns of ``refined.cif``.

An output written with the deposit's polymer categories carries the deposit's ``label_seq_id``
and ``label_asym_id`` and the input's numbering in the ``auth_`` columns. Atomworks loads the
label columns into ``res_id`` and ``chain_id``, so without the author relabeling a selection such
as ``chain A and resi 326-339`` would land on the deposit's numbering and the atom pairing with
the reference would shift or empty out. The fixture here is the 1VME reference with its label
columns moved away from the author columns.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from atomworks.io.utils.io_utils import load_any
from biotite.structure.io.pdbx import CIFFile
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.utils.atom_array_utils import filter_to_common_atoms, use_author_numbering
from sampleworks.utils.structure_utils import get_reference_atomarraystack

from tests.eval.conftest import _REAL_CIF


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "eval"
_LABEL_OFFSET = 1000
_LABEL_CHAIN = "Z"
_AUTHOR_FIELDS = ["auth_seq_id", "auth_asym_id"]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rmsd_script():
    return _load_script("rmsd_evaluation_script")


@pytest.fixture
def rscc_script():
    return _load_script("rscc_grid_search_script")


def write_deposit_labeled_cif(source: Path, destination: Path) -> Path:
    """Copy ``source`` with ``label_seq_id`` offset and ``label_asym_id`` relettered.

    The ``auth_`` columns are untouched, which is the shape of an output whose polymer
    categories were carried from a deposit whose numbering differs from the input's.
    """
    cif = CIFFile.read(source)
    atom_site = cif.block["atom_site"]
    label_seq_id = atom_site["label_seq_id"].as_array(str)
    # Built as a list: a fixed-width string array would truncate the longer numbers.
    shifted = np.array(
        [
            str(int(value) + _LABEL_OFFSET) if value.lstrip("-").isdigit() else value
            for value in label_seq_id
        ]
    )
    atom_site["label_seq_id"] = shifted
    atom_site["label_asym_id"] = np.full(len(label_seq_id), _LABEL_CHAIN)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cif.write(destination)
    return destination


def _relabel_every_refined_cif(fx) -> list[Path]:
    """Replace each trial's ``refined.cif`` link with the deposit-labeled copy."""
    paths = sorted(Path(fx.grid_search_results_path).rglob("refined.cif"))
    assert paths
    for path in paths:
        path.unlink()
        write_deposit_labeled_cif(_REAL_CIF, path)
    return paths


def test_deposit_labels_pair_with_the_reference_after_relabeling(tmp_path: Path):
    """Loaded as the evaluators load them, every atom of the relabeled file pairs with the input."""
    relabeled = write_deposit_labeled_cif(_REAL_CIF, tmp_path / "refined.cif")

    reference = use_author_numbering(load_any(_REAL_CIF, extra_fields=_AUTHOR_FIELDS))
    predicted = use_author_numbering(load_any(relabeled, extra_fields=_AUTHOR_FIELDS))
    ref_common, pred_common = filter_to_common_atoms(reference, predicted)

    assert ref_common.array_length() == reference.array_length()
    assert pred_common.array_length() == predicted.array_length()
    assert set(np.asarray(predicted.chain_id)) == set(np.asarray(reference.chain_id))


def test_reference_loader_returns_author_numbering(tmp_path: Path):
    """A deposit-labeled reference comes back from the shared loader on the author numbering."""
    write_deposit_labeled_cif(_REAL_CIF, tmp_path / "reference.cif")
    config = ProteinConfig(
        protein="1vme",
        base_map_dir=tmp_path,
        selection=["chain A"],
        resolution=2.0,
        map_pattern="{occ_str}.ccp4",
        structure_pattern="reference.cif",
    )

    _, stack = get_reference_atomarraystack(config, {"A": 0.5, "B": 0.5})

    assert stack is not None
    original = use_author_numbering(load_any(_REAL_CIF, extra_fields=_AUTHOR_FIELDS))
    assert set(np.asarray(stack.chain_id)) == set(np.asarray(original.chain_id))
    assert stack.res_id.min() == original.res_id.min()
    assert stack.res_id.max() == original.res_id.max()


def test_rmsd_script_scores_deposit_labeled_output(rscc_fixture_factory, rmsd_script):
    """The RMSD evaluator gives the same answer on the relabeled file as on the reference itself."""
    fx = rscc_fixture_factory(
        n_groups=1, trials_per_group=1, selections=("chain A and resi 326-339",)
    )
    _relabel_every_refined_cif(fx)
    args = argparse.Namespace(
        grid_search_results_path=fx.grid_search_results_path,
        grid_search_inputs_path=fx.grid_search_inputs_path,
        protein_configs_csv=fx.protein_configs_csv,
        occupancies=[0.5],
        target_filename="refined.cif",
        depth=4,
        n_jobs=1,
    )

    rmsd_script.main(args)

    df = pd.read_csv(fx.grid_search_results_path / "min_altloc_rmsd_results.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["min_rmsd_to_A"] == pytest.approx(0.0, abs=1e-4)
    assert row["min_rmsd_to_B"] > 0.1


@pytest.mark.slow
def test_rscc_script_scores_deposit_labeled_output(rscc_fixture_factory, rscc_script):
    """RSCC on the relabeled file is the same strong self-correlation as on the reference itself."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)
    _relabel_every_refined_cif(fx)

    rscc_script.main(fx)

    df = pd.read_csv(fx.grid_search_results_path / "rscc_results.csv")
    assert len(df) == len(fx.selections)
    assert df["error"].isna().all(), f"unexpected errors: {df['error'].tolist()}"
    assert (df["rscc"] > 0.89).all(), f"expected strong self-correlation, got {df['rscc'].tolist()}"
