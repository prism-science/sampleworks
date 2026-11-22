"""
The evaluators, the altloc scripts and the synthetic generators select and pair atoms on the
author columns of the files they load.

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
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.synthetic.synthetic_utils import load_structure_for_synthetic_loss
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    load_conformer_pairs,
    load_structure_with_altlocs,
    load_with_author_numbering,
    use_author_numbering,
)
from sampleworks.utils.cif_utils import find_altloc_selections
from sampleworks.utils.structure_utils import get_reference_atomarraystack

from tests.eval.conftest import _REAL_CIF
from tests.script_loader import load_script


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "eval"
# The PDB entry's own mmCIF: label and author numbering differ, and it has waters and ligands.
_DEPOSITED_CIF = _REAL_CIF.parent / "1vme_rcsb.cif"
_PDB_FILE = _REAL_CIF.parents[1] / "9BN8" / "9BN8_single_001.pdb"
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


@pytest.fixture
def lddt_script():
    """The lDDT evaluation script, imported by path."""
    return load_script(_SCRIPTS / "lddt_evaluation_script.py")


def read_atom_site_column(path: Path, column: str) -> np.ndarray:
    """Read one ``atom_site`` column of an mmCIF file with biotite alone, as written.

    Parameters
    ----------
    path : Path
        mmCIF file.
    column : str
        Column name without the category, e.g. ``"auth_seq_id"``.

    Returns
    -------
    np.ndarray
        The column as strings, one per ``atom_site`` row, shape ``(n_rows,)``.
    """
    return CIFFile.read(path).block["atom_site"][column].as_array(str)


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


def _eval_args(fx) -> argparse.Namespace:
    """Arguments the RMSD and lDDT scripts take, for one fixture tree."""
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
    """A label-moved reference comes back from the shared loader on its author columns."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "reference.cif", "label")
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
    np.testing.assert_array_equal(
        reference_stack.res_id, read_atom_site_column(label_moved, "auth_seq_id").astype(int)
    )
    np.testing.assert_array_equal(
        reference_stack.chain_id, read_atom_site_column(label_moved, "auth_asym_id")
    )


def test_loader_keeps_both_numberings_of_a_deposited_file():
    """The author columns become res_id and chain_id; the label columns stay as annotations."""
    atom_array = load_with_author_numbering(_DEPOSITED_CIF)

    np.testing.assert_array_equal(
        atom_array.res_id, read_atom_site_column(_DEPOSITED_CIF, "auth_seq_id").astype(int)
    )
    np.testing.assert_array_equal(
        atom_array.chain_id, read_atom_site_column(_DEPOSITED_CIF, "auth_asym_id")
    )
    np.testing.assert_array_equal(
        atom_array.get_annotation("label_asym_id"),
        read_atom_site_column(_DEPOSITED_CIF, "label_asym_id"),
    )
    written_label_seq_id = read_atom_site_column(_DEPOSITED_CIF, "label_seq_id")
    in_polymer = written_label_seq_id != "."
    kept_label_seq_id = atom_array.get_annotation("label_seq_id")
    np.testing.assert_array_equal(
        kept_label_seq_id[in_polymer], written_label_seq_id[in_polymer].astype(int)
    )
    # biotite loads the ``.`` of an atom outside a polymer as -1.
    assert in_polymer.any() and not in_polymer.all()
    assert (kept_label_seq_id[~in_polymer] == -1).all()
    assert not np.array_equal(atom_array.res_id[in_polymer], kept_label_seq_id[in_polymer])


def test_pdb_file_keeps_its_own_numbering():
    """A PDB file has one numbering, so the loader returns it with nothing relabeled."""
    atom_array = load_with_author_numbering(_PDB_FILE)

    as_written = PDBFile.read(_PDB_FILE).get_structure(model=1, altloc="all")
    np.testing.assert_array_equal(atom_array.res_id, as_written.res_id)
    np.testing.assert_array_equal(atom_array.chain_id, as_written.chain_id)
    assert "label_seq_id" not in atom_array.get_annotation_categories()


def test_rscc_parse_numbers_label_moved_file_by_author_columns(tmp_path: Path, rscc_script):
    """The RSCC script's parse puts a label-moved file on its unmoved author columns."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "refined.cif", "label")

    atom_array = rscc_script.parse_on_author_numbering(label_moved)

    assert set(atom_array.chain_id) == set(read_atom_site_column(_REAL_CIF, "auth_asym_id"))
    assert set(atom_array.get_annotation("label_asym_id")) == {_MOVED_CHAIN}
    assert set(atom_array.get_annotation("label_seq_id") - atom_array.res_id) == {_SEQ_ID_OFFSET}


def test_rscc_parse_accepts_a_pdb_reference(rscc_script):
    """A PDB reference has no author columns to load and parses on the numbering it has."""
    atom_array = rscc_script.parse_on_author_numbering(_PDB_FILE)

    as_written = PDBFile.read(_PDB_FILE).get_structure(model=1)
    polymer_res_ids = set(as_written.res_id[~as_written.hetero])
    assert polymer_res_ids <= set(atom_array.res_id)
    assert "label_seq_id" not in atom_array.get_annotation_categories()


def test_altloc_loader_returns_author_numbering(tmp_path: Path):
    """The altloc loader gives a label-moved file the ids of the unmoved file."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "moved.cif", "label")

    moved_atom_array = load_structure_with_altlocs(label_moved)
    unmoved_atom_array = load_structure_with_altlocs(_REAL_CIF)

    np.testing.assert_array_equal(moved_atom_array.chain_id, unmoved_atom_array.chain_id)
    np.testing.assert_array_equal(moved_atom_array.res_id, unmoved_atom_array.res_id)


def test_synthetic_loader_selects_on_author_numbering(tmp_path: Path):
    """A selection in author numbering picks the same atoms from a label-moved file."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "moved.cif", "label")

    moved_selected = load_structure_for_synthetic_loss(
        label_moved, "default", [], selection=_SELECTION
    )
    unmoved_selected = load_structure_for_synthetic_loss(
        _REAL_CIF, "default", [], selection=_SELECTION
    )

    assert moved_selected is not None
    assert unmoved_selected is not None
    assert unmoved_selected.array_length() > 0
    np.testing.assert_array_equal(moved_selected.res_id, unmoved_selected.res_id)
    np.testing.assert_array_equal(moved_selected.atom_name, unmoved_selected.atom_name)


def test_altloc_selections_are_in_author_numbering(tmp_path: Path):
    """find_altloc_selections gives a label-moved file the selections of the unmoved file."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "moved.cif", "label")

    moved_spans, moved_unions = find_altloc_selections(label_moved)
    unmoved_spans, unmoved_unions = find_altloc_selections(_REAL_CIF)

    assert unmoved_spans
    assert unmoved_unions
    assert moved_spans == unmoved_spans
    assert moved_unions == unmoved_unions


def test_conformer_pairs_are_on_author_numbering(tmp_path: Path):
    """The conformer pairs of a label-moved file carry the ids of the unmoved file."""
    label_moved = write_cif_with_moved_ids(_REAL_CIF, tmp_path / "moved.cif", "label")

    moved_atom_array, moved_conformer_ids, moved_pairs = load_conformer_pairs(label_moved)
    unmoved_atom_array, unmoved_conformer_ids, unmoved_pairs = load_conformer_pairs(_REAL_CIF)

    assert moved_conformer_ids == unmoved_conformer_ids
    assert unmoved_pairs
    np.testing.assert_array_equal(moved_atom_array.res_id, unmoved_atom_array.res_id)
    for pair, (unmoved_first, _) in unmoved_pairs.items():
        moved_first, _ = moved_pairs[pair]
        np.testing.assert_array_equal(moved_first.chain_id, unmoved_first.chain_id)
        np.testing.assert_array_equal(moved_first.res_id, unmoved_first.res_id)


def test_rmsd_script_scores_label_moved_output(rscc_fixture_factory, rmsd_script):
    """The RMSD evaluator gives the same answer on the label-moved file as on the reference."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=(_SELECTION,))
    _replace_every_refined_cif(fx, "label")

    rmsd_script.main(_eval_args(fx))

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
        rmsd_script.main(_eval_args(fx))


def test_lddt_script_scores_label_moved_output(rscc_fixture_factory, lddt_script):
    """The lDDT evaluator assigns the label-moved file, a copy of altloc A, to altloc A."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=(_SELECTION,))
    _replace_every_refined_cif(fx, "label")

    lddt_script.main(_eval_args(fx))

    df = pd.read_csv(fx.grid_search_results_path / "lddt_results.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert pd.isna(row["error"]), row["error"]
    # Fraction of output models closest to altloc A, then to altloc B, as numpy prints them.
    occupancies = np.array(row["occupancies"].strip("[]").split(), dtype=float)
    np.testing.assert_allclose(occupancies, [1.0, 0.0])
    assert row["avg_silhouette_to_ref"] == pytest.approx(1.0)


def test_lddt_script_cannot_score_author_moved_output(rscc_fixture_factory, lddt_script):
    """With the author columns moved instead, the selection finds nothing to score.

    The lDDT script records a failed trial as an error row instead of raising. An evaluator
    reading the label columns would fail this way on the label-moved file and score this one.
    """
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=(_SELECTION,))
    _replace_every_refined_cif(fx, "auth")

    lddt_script.main(_eval_args(fx))

    df = pd.read_csv(fx.grid_search_results_path / "lddt_results.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert pd.notna(row["error"])
    assert np.isnan(row["avg_silhouette_to_ref"])


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
