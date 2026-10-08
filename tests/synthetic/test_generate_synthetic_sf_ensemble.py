"""Behavior tests for synthetic ensemble structure-factor generation."""

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch
from sampleworks.synthetic.generate_synthetic_sf import (
    _process_single_row as process_single_structure,
    BatchRowForMTZ,
)
from sampleworks.synthetic.generate_synthetic_sf_ensemble import (
    _process_single_row as process_single_ensemble,
    process_batch,
)
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs


# Toy two-conformer ensemble: one glycine on a monoclinic cell, the second conformer
# shifted 0.2 A along x.
CELL = (31.0, 32.0, 33.0, 90.0, 91.0, 90.0)
OFFSETS = [0.0, 0.2]
OCCUPANCIES = [0.4, 0.6]
SHARED_SETTINGS: dict[str, Any] = {
    "resolution": 4.0,
    "scattering_factor_mode": "xray",
    "test_fraction": 0.1,
    "seed": 7,
    "device": torch.device("cpu"),
    "strip_hydrogens": True,
    "b_factor": 20.0,
}


def _write_structure(path: Path, models: list[list[tuple[float, float, str]]]) -> None:
    """Write a one-residue crystallographic structure with one removable hydrogen.

    Parameters
    ----------
    path
        Output PDB path.
    models
        One entry per model, each a list of ``(x_offset, occupancy, altloc)`` conformers
        that add a full copy of the residue's atoms shifted by ``x_offset`` along x. A
        blank altloc writes a plain single-conformer model.

    Notes
    -----
    This function writes a PDB file at ``path``.
    """
    structure = gemmi.Structure()
    for model_index, conformers in enumerate(models, start=1):
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.seqid = gemmi.SeqId(1, " ")
        residue.het_flag = "A"
        for x_offset, occupancy, altloc in conformers:
            for atom_name, element, x_coordinate in (
                ("N", "N", 1.0),
                ("H", "H", 1.5),
                ("CA", "C", 2.0),
                ("C", "C", 3.0),
                ("O", "O", 4.0),
            ):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(element)
                atom.pos = gemmi.Position(x_coordinate + x_offset, 2.0, 3.0)
                atom.b_iso = 15.0
                atom.occ = occupancy
                atom.altloc = altloc
                residue.add_atom(atom)
        chain = gemmi.Chain("A")
        chain.add_residue(residue)
        model = gemmi.Model(str(model_index))
        model.add_chain(chain)
        structure.add_model(model)
    structure.cell = gemmi.UnitCell(*CELL)
    structure.spacegroup_hm = "P 1 21 1"
    structure.write_pdb(str(path))


def _write_ensemble(path: Path) -> None:
    """Write the toy ensemble as a multi-model file, one conformer per model.

    Parameters
    ----------
    path
        Output PDB path.

    Notes
    -----
    This function writes a PDB file at ``path``.
    """
    _write_structure(path, [[(offset, 1.0, " ")] for offset in OFFSETS])


def test_summed_models_match_merged_altloc_structure(tmp_path: Path) -> None:
    """Verify the per-model sum reproduces the merged altloc structure's MTZ.

    This is the assumption the script rests on: Fprotein is linear in occupancy-weighted
    atoms, so splitting altlocs into models and summing loses nothing.

    Parameters
    ----------
    tmp_path
        Temporary directory for structures and generated outputs.
    """
    _write_structure(tmp_path / "merged.pdb", [list(zip(OFFSETS, OCCUPANCIES, "AB", strict=True))])
    _write_ensemble(tmp_path / "ensemble.pdb")

    process_single_structure(
        row=BatchRowForMTZ(filename="merged.pdb", mtzfile="merged.mtz"),
        base_dir=tmp_path,
        output_dir=tmp_path,
        occupancy_mode="default",
        **SHARED_SETTINGS,
    )
    process_single_ensemble(
        row=BatchRowForMTZ(
            filename="ensemble.pdb", occupancy_values=OCCUPANCIES, mtzfile="ensemble.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        **SHARED_SETTINGS,
    )

    merged = rs.read_mtz(str(tmp_path / "merged.mtz")).sort_index()
    ensemble = rs.read_mtz(str(tmp_path / "ensemble.mtz")).sort_index()
    assert list(ensemble.columns) == list(merged.columns)
    assert ensemble.index.equals(merged.index)
    assert ensemble.cell.parameters == merged.cell.parameters
    assert ensemble.spacegroup.hm == merged.spacegroup.hm
    # R-free flags depend only on the reflection list and seed, so they must match exactly.
    flag_columns = merged.columns.difference(["Fprotein", "SIGFprotein", "PHIFprotein"])
    assert ensemble[flag_columns].equals(merged[flag_columns])
    # Compare complex values so phases of near-zero amplitudes cannot fail on wrap-around.
    merged_f = merged.to_structurefactor("Fprotein", "PHIFprotein").to_numpy()
    ensemble_f = ensemble.to_structurefactor("Fprotein", "PHIFprotein").to_numpy()
    np.testing.assert_allclose(ensemble_f, merged_f, atol=1e-4 * np.abs(merged_f).max())


def _complex_column(dataset: rs.DataSet, label: str) -> np.ndarray:
    """Return one structure-factor set of an MTZ dataset as complex values.

    Parameters
    ----------
    dataset
        Dataset holding ``F{label}`` and ``PHIF{label}`` columns.
    label
        Structure-factor label, e.g. ``"protein"`` or ``"total"``.

    Returns
    -------
    np.ndarray
        Complex structure factors ``[n_hkl]``. Comparing these rather than phases keeps
        near-zero amplitudes from failing on phase wrap-around.
    """
    return dataset.to_structurefactor(f"F{label}", f"PHIF{label}").to_numpy()


@pytest.mark.parametrize("bulk_solvent", ["combined", "per_conformer"])
def test_identical_models_match_single_structure_with_solvent(
    tmp_path: Path, bulk_solvent: str
) -> None:
    """Verify both solvent modes reproduce the single-structure Ftotal for a degenerate ensemble.

    Two identical models at populations 0.4/0.6 are one structure, so the combined mask,
    the population-weighted per-model masks, and the single-structure mask must agree.

    Parameters
    ----------
    tmp_path
        Temporary directory for structures and generated outputs.
    bulk_solvent
        Bulk-solvent mode under test.
    """
    _write_structure(tmp_path / "single.pdb", [[(0.0, 1.0, " ")]])
    _write_structure(tmp_path / "ensemble.pdb", [[(0.0, 1.0, " ")]] * 2)

    process_single_structure(
        row=BatchRowForMTZ(filename="single.pdb", mtzfile="single.mtz"),
        base_dir=tmp_path,
        output_dir=tmp_path,
        occupancy_mode="default",
        simulate_solvent_and_scale=True,
        **SHARED_SETTINGS,
    )
    process_single_ensemble(
        row=BatchRowForMTZ(
            filename="ensemble.pdb", occupancy_values=OCCUPANCIES, mtzfile="ensemble.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        bulk_solvent=bulk_solvent,
        **SHARED_SETTINGS,
    )

    single = rs.read_mtz(str(tmp_path / "single.mtz")).sort_index()
    ensemble = rs.read_mtz(str(tmp_path / "ensemble.mtz")).sort_index()
    assert list(ensemble.columns) == list(single.columns)
    assert ensemble.index.equals(single.index)
    for label in ("protein", "total"):
        single_f = _complex_column(single, label)
        np.testing.assert_allclose(
            _complex_column(ensemble, label), single_f, atol=1e-4 * np.abs(single_f).max()
        )


def test_solvent_modes_leave_fprotein_unchanged(tmp_path: Path) -> None:
    """Verify bulk solvent only adds an Ftotal set beside an unchanged Fprotein set.

    Parameters
    ----------
    tmp_path
        Temporary directory for the ensemble and generated outputs.
    """
    _write_ensemble(tmp_path / "ensemble.pdb")

    datasets = {}
    for bulk_solvent in ("off", "combined", "per_conformer"):
        process_single_ensemble(
            row=BatchRowForMTZ(
                filename="ensemble.pdb",
                occupancy_values=OCCUPANCIES,
                mtzfile=f"{bulk_solvent}.mtz",
            ),
            base_dir=tmp_path,
            output_dir=tmp_path,
            bulk_solvent=bulk_solvent,
            **SHARED_SETTINGS,
        )
        datasets[bulk_solvent] = rs.read_mtz(str(tmp_path / f"{bulk_solvent}.mtz")).sort_index()

    assert "Ftotal" not in datasets["off"].columns
    for bulk_solvent in ("combined", "per_conformer"):
        dataset = datasets[bulk_solvent]
        assert {"Ftotal", "SIGFtotal", "PHIFtotal"} <= set(dataset.columns)
        assert np.isfinite(dataset["Ftotal"].to_numpy()).all()
        np.testing.assert_array_equal(
            _complex_column(dataset, "protein"), _complex_column(datasets["off"], "protein")
        )


def test_batch_csv_matches_single_ensemble_mode(tmp_path: Path) -> None:
    """Verify a CSV row naming a multi-model file writes the single-mode MTZ.

    Parameters
    ----------
    tmp_path
        Temporary directory for the ensemble, the CSV, and generated outputs.
    """
    _write_ensemble(tmp_path / "ensemble.pdb")
    csv_path = tmp_path / "ensembles.csv"
    csv_path.write_text("filename,occupancy_values,mtzfile\nensemble.pdb,0.4:0.6,ab.mtz\n")
    batch_dir = tmp_path / "batch"

    process_batch(
        csv_path=csv_path, base_dir=tmp_path, output_dir=batch_dir, n_jobs=1, **SHARED_SETTINGS
    )
    process_single_ensemble(
        row=BatchRowForMTZ(filename="ensemble.pdb", occupancy_values=OCCUPANCIES, mtzfile="ab.mtz"),
        base_dir=tmp_path,
        output_dir=tmp_path,
        **SHARED_SETTINGS,
    )

    assert (batch_dir / "ab.mtz").read_bytes() == (tmp_path / "ab.mtz").read_bytes()


def test_occupancy_count_mismatch_writes_no_mtz(tmp_path: Path) -> None:
    """Verify an ensemble whose occupancies do not match its model count is refused.

    Parameters
    ----------
    tmp_path
        Temporary directory for the ensemble and generated outputs.
    """
    _write_ensemble(tmp_path / "ensemble.pdb")

    process_single_ensemble(
        row=BatchRowForMTZ(
            filename="ensemble.pdb", occupancy_values=[0.2, 0.3, 0.5], mtzfile="x.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        **SHARED_SETTINGS,
    )

    assert not (tmp_path / "x.mtz").exists()


def test_saved_structure_keeps_one_model_per_conformer(tmp_path: Path) -> None:
    """Verify the saved SF input holds every processed model at occupancy 1.0.

    Parameters
    ----------
    tmp_path
        Temporary directory for the ensemble and generated outputs.
    """
    _write_ensemble(tmp_path / "ensemble.pdb")

    process_single_ensemble(
        row=BatchRowForMTZ(
            filename="ensemble.pdb", occupancy_values=OCCUPANCIES, mtzfile="out/ensemble.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        save_structure=True,
        **SHARED_SETTINGS,
    )

    saved_path = tmp_path / "out" / "ensemble_sf_input.cif"
    assert len(gemmi.read_structure(str(saved_path))) == len(OFFSETS)
    for model, offset in enumerate(OFFSETS):
        saved = load_structure_with_altlocs(saved_path, model=model)
        assert list(saved.atom_name) == ["N", "CA", "C", "O"]  # hydrogen stripped
        np.testing.assert_allclose(saved.coord[:, 0], np.array([1.0, 2.0, 3.0, 4.0]) + offset)
        np.testing.assert_array_equal(saved.occupancy, 1.0)
        np.testing.assert_array_equal(saved.b_factor, 20.0)
