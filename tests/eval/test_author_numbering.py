"""
The evaluators select and pair atoms on the author columns of the files they load.

Atomworks loads the label columns into ``res_id`` and ``chain_id``. An output whose label
columns were rewritten to a deposit's numbering no longer shares them with the reference, so a
selection such as ``chain A and resi 326-339`` would miss and the atom pairing would shift or
empty out. The fixtures here are the 1VME reference with one pair of id columns, label or
author, moved away from the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from atomworks.io.utils.io_utils import load_any
from biotite.structure.io.pdbx import CIFFile
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.utils.atom_array_utils import filter_to_common_atoms, use_author_numbering
from sampleworks.utils.structure_utils import get_reference_atomarraystack

from tests.eval.conftest import _REAL_CIF
from tests.script_loader import load_script


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "eval"
_SEQ_ID_OFFSET = 1000
_MOVED_CHAIN = "Z"
_AUTHOR_FIELDS = ["auth_seq_id", "auth_asym_id"]
_SELECTION = "chain A and resi 326-339"


@pytest.fixture
def rmsd_script():
    """The RMSD evaluation script, imported by path."""
    return load_script(_SCRIPTS / "rmsd_evaluation_script.py")


@pytest.fixture
def rscc_script():
    """The RSCC evaluation script, imported by path."""
    return load_script(_SCRIPTS / "rscc_grid_search_script.py")


def write_cif_with_moved_ids(
    source: Path, destination: Path, id_columns: Literal["label", "auth"]
) -> Path:
    """Copy ``source`` with one pair of id columns moved away from the other.

    ``"label"`` gives the shape of an output whose label columns carry a deposit's numbering
    while its author columns still match the reference. ``"auth"`` is the opposite file, whose
    author columns match nothing in the reference.

    Parameters
    ----------
    source : Path
        mmCIF file whose ``{id_columns}_seq_id`` values are all integers (insertion codes and
        ``.`` placeholders are not handled) and which has no chain named ``_MOVED_CHAIN``.
    destination : Path
        Where the copy is written.
    id_columns : {"label", "auth"}
        The pair to move: its ``_seq_id`` is offset by ``_SEQ_ID_OFFSET`` and its ``_asym_id``
        is set to ``_MOVED_CHAIN``.

    Returns
    -------
    Path
        ``destination``.
    """
    cif = CIFFile.read(source)
    atom_site = cif.block["atom_site"]
    seq_id = atom_site[f"{id_columns}_seq_id"].as_array(int)
    asym_id = atom_site[f"{id_columns}_asym_id"].as_array(str)
    assert _MOVED_CHAIN not in asym_id, f"{source} already has a chain {_MOVED_CHAIN}"
    atom_site[f"{id_columns}_seq_id"] = (seq_id + _SEQ_ID_OFFSET).astype(str)
    atom_site[f"{id_columns}_asym_id"] = np.full(len(asym_id), _MOVED_CHAIN)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cif.write(destination)
    return destination


def _replace_every_refined_cif(fx, id_columns: Literal["label", "auth"]) -> None:
    """Replace each trial's ``refined.cif`` link with a copy whose ``id_columns`` are moved."""
    paths = sorted(Path(fx.grid_search_results_path).rglob("refined.cif"))
    assert paths
    for path in paths:
        path.unlink()
        write_cif_with_moved_ids(_REAL_CIF, path, id_columns)


def _rmsd_args(fx) -> argparse.Namespace:
    return argparse.Namespace(
        grid_search_results_path=fx.grid_search_results_path,
        grid_search_inputs_path=fx.grid_search_inputs_path,
        protein_configs_csv=fx.protein_configs_csv,
        occupancies=[0.5],
        target_filename="refined.cif",
        depth=4,
        n_jobs=1,
    )


def test_label_moved_output_pairs_with_the_reference_on_author_numbering(tmp_path: Path):
    """Read on the author columns, every atom of the label-moved file pairs with the reference."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "refined.cif", "label")

    reference_atom_array = use_author_numbering(load_any(_REAL_CIF, extra_fields=_AUTHOR_FIELDS))
    predicted_atom_array = use_author_numbering(load_any(label_moved, extra_fields=_AUTHOR_FIELDS))
    reference_common, predicted_common = filter_to_common_atoms(
        reference_atom_array, predicted_atom_array
    )

    assert reference_common.array_length() == reference_atom_array.array_length()
    assert predicted_common.array_length() == predicted_atom_array.array_length()
    assert set(np.asarray(predicted_atom_array.chain_id)) == set(
        np.asarray(reference_atom_array.chain_id)
    )


def test_label_moved_output_shares_no_atom_with_the_reference_on_label_numbering(tmp_path: Path):
    """Without the relabeling, the same file has no atom id in common with the reference."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "refined.cif", "label")

    reference_atom_array = load_any(_REAL_CIF, extra_fields=_AUTHOR_FIELDS)
    predicted_atom_array = load_any(label_moved, extra_fields=_AUTHOR_FIELDS)

    with pytest.raises(RuntimeError, match="No common atoms"):
        filter_to_common_atoms(reference_atom_array, predicted_atom_array)


def test_reference_loader_returns_author_numbering(tmp_path: Path):
    """A label-moved reference comes back from the shared loader on the author numbering."""
    write_cif_with_moved_ids(_REAL_CIF, tmp_path / "reference.cif", "label")
    config = ProteinConfig(
        protein="1vme",
        base_map_dir=tmp_path,
        selection=["chain A"],
        resolution=2.0,
        map_pattern="{occ_str}.ccp4",
        structure_pattern="reference.cif",
    )

    _, reference_stack = get_reference_atomarraystack(config, {"A": 0.5, "B": 0.5})

    assert reference_stack is not None
    unmoved = use_author_numbering(load_any(_REAL_CIF, extra_fields=_AUTHOR_FIELDS))
    assert set(np.asarray(reference_stack.chain_id)) == set(np.asarray(unmoved.chain_id))
    assert reference_stack.res_id.min() == unmoved.res_id.min()
    assert reference_stack.res_id.max() == unmoved.res_id.max()


def test_rmsd_script_scores_label_moved_output(rscc_fixture_factory, rmsd_script):
    """The RMSD evaluator gives the same answer on the label-moved file as on the reference."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=(_SELECTION,))
    _replace_every_refined_cif(fx, "label")

    rmsd_script.main(_rmsd_args(fx))

    df = pd.read_csv(fx.grid_search_results_path / "min_altloc_rmsd_results.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["min_rmsd_to_A"] == pytest.approx(0.0, abs=1e-4)
    assert row["min_rmsd_to_B"] > 0.1


def test_rmsd_script_cannot_pair_author_moved_output(rscc_fixture_factory, rmsd_script):
    """With the author columns moved instead, the evaluator finds no atom to pair.

    An evaluator reading the label columns would fail this way on the label-moved file and
    score this one, so the two tests together pin which columns are read.
    """
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=(_SELECTION,))
    _replace_every_refined_cif(fx, "auth")

    with pytest.raises(RuntimeError, match="No common atoms"):
        rmsd_script.main(_rmsd_args(fx))


@pytest.mark.slow
def test_rscc_script_scores_label_moved_output(rscc_fixture_factory, rscc_script):
    """RSCC on the label-moved file is the same strong self-correlation as on the reference."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)
    _replace_every_refined_cif(fx, "label")

    rscc_script.main(fx)

    df = pd.read_csv(fx.grid_search_results_path / "rscc_results.csv")
    assert len(df) == len(fx.selections)
    assert df["error"].isna().all(), f"unexpected errors: {df['error'].tolist()}"
    assert (df["rscc"] > 0.89).all(), f"expected strong self-correlation, got {df['rscc'].tolist()}"
