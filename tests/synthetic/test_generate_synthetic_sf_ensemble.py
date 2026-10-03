"""Behavior tests for synthetic ensemble structure-factor generation."""

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from sampleworks.synthetic.generate_synthetic_sf import (
    _process_single_row as process_single_structure,
    BatchRowForMTZ,
)
from sampleworks.synthetic.generate_synthetic_sf_ensemble import (
    _process_single_row as process_single_ensemble,
    EnsembleBatchRowForMTZ,
    process_batch,
)


# Toy two-conformer ensemble: one glycine on a monoclinic cell, the second conformer
# shifted 0.2 A along x.
CELL = (31.0, 32.0, 33.0, 90.0, 91.0, 90.0)
CONFORMER_FILES = ["conf_a.pdb", "conf_b.pdb"]
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


def _write_structure(
    path: Path,
    conformers: list[tuple[float, float, str]],
    cell: tuple[float, ...] = CELL,
) -> None:
    """Write a one-residue crystallographic structure with one removable hydrogen.

    Parameters
    ----------
    path
        Output PDB path.
    conformers
        One ``(x_offset, occupancy, altloc)`` per conformer, each adding a full copy of the
        residue's atoms shifted by ``x_offset`` along x. A blank altloc writes a plain
        single-conformer structure.
    cell
        Unit cell parameters (a, b, c, alpha, beta, gamma).

    Notes
    -----
    This function writes a PDB file at ``path``.
    """
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
    model = gemmi.Model("1")
    model.add_chain(chain)
    structure = gemmi.Structure()
    structure.add_model(model)
    structure.cell = gemmi.UnitCell(*cell)
    structure.spacegroup_hm = "P 1 21 1"
    structure.write_pdb(str(path))


def test_summed_conformers_match_merged_altloc_structure(tmp_path: Path) -> None:
    """Verify the per-conformer sum reproduces the merged altloc structure's MTZ.

    This is the assumption the script rests on: Fprotein is linear in occupancy-weighted
    atoms, so splitting altlocs into files and summing loses nothing.

    Parameters
    ----------
    tmp_path
        Temporary directory for structures and generated outputs.
    """
    _write_structure(
        tmp_path / "merged.pdb",
        list(zip(OFFSETS, OCCUPANCIES, "AB", strict=True)),
    )
    for filename, offset in zip(CONFORMER_FILES, OFFSETS, strict=True):
        _write_structure(tmp_path / filename, [(offset, 1.0, " ")])

    process_single_structure(
        row=BatchRowForMTZ(filename="merged.pdb", mtzfile="merged.mtz"),
        base_dir=tmp_path,
        output_dir=tmp_path,
        occupancy_mode="default",
        **SHARED_SETTINGS,
    )
    process_single_ensemble(
        row=EnsembleBatchRowForMTZ(
            filenames=CONFORMER_FILES,
            occupancy_values=OCCUPANCIES,
            mtzfile="ensemble.mtz",
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


def test_batch_csv_matches_single_ensemble_mode(tmp_path: Path) -> None:
    """Verify a CSV row with colon-separated conformers writes the single-mode MTZ.

    Parameters
    ----------
    tmp_path
        Temporary directory for conformers, the CSV, and generated outputs.
    """
    for filename, offset in zip(CONFORMER_FILES, OFFSETS, strict=True):
        _write_structure(tmp_path / filename, [(offset, 1.0, " ")])
    csv_path = tmp_path / "ensembles.csv"
    csv_path.write_text(
        "filenames,occupancy_values,mtzfile\nconf_a.pdb:conf_b.pdb,0.4:0.6,ab.mtz\n"
    )
    batch_dir = tmp_path / "batch"

    process_batch(
        csv_path=csv_path, base_dir=tmp_path, output_dir=batch_dir, n_jobs=1, **SHARED_SETTINGS
    )
    process_single_ensemble(
        row=EnsembleBatchRowForMTZ(
            filenames=CONFORMER_FILES, occupancy_values=OCCUPANCIES, mtzfile="ab.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        **SHARED_SETTINGS,
    )

    assert (batch_dir / "ab.mtz").read_bytes() == (tmp_path / "ab.mtz").read_bytes()


def test_conformers_on_different_cells_write_no_mtz(tmp_path: Path) -> None:
    """Verify conformers that disagree on the unit cell are refused rather than summed.

    Parameters
    ----------
    tmp_path
        Temporary directory for conformers and generated outputs.
    """
    _write_structure(tmp_path / CONFORMER_FILES[0], [(OFFSETS[0], 1.0, " ")])
    _write_structure(
        tmp_path / CONFORMER_FILES[1], [(OFFSETS[1], 1.0, " ")], cell=(40.0, *CELL[1:])
    )

    process_single_ensemble(
        row=EnsembleBatchRowForMTZ(
            filenames=CONFORMER_FILES, occupancy_values=OCCUPANCIES, mtzfile="x.mtz"
        ),
        base_dir=tmp_path,
        output_dir=tmp_path,
        **SHARED_SETTINGS,
    )

    assert not (tmp_path / "x.mtz").exists()
