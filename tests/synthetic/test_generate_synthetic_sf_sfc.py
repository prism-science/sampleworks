"""Behavior tests for synthetic structure-factor generation from multi-conformer and multi-model."""

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch
from sampleworks.synthetic.generate_synthetic_sf_sfc import (
    _process_single_row,
    load_models,
    process_batch,
)
from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs


# Toy two-conformer ensemble: one glycine on a monoclinic cell whose N/H/CA are shared and
# whose C/O differ in conformation, with the second conformer's shifted 0.2 A along x.
CELL = (31.0, 32.0, 33.0, 90.0, 91.0, 90.0)
SPACEGROUP = "P 1 21 1"
OFFSETS = [0.0, 0.2]
OCCUPANCIES = [0.4, 0.6]
ALTLOC_IDS = ["A", "B"]
BLANK_ALTLOC_ID = " "
SHARED_SETTINGS: dict[str, Any] = {
    "resolution": 4.0,
    "scattering_factor_mode": "xray",
    "test_fraction": 0.1,
    "seed": 7,
    "device": torch.device("cpu"),
    "strip_hydrogens": True,
    "b_factor": 20.0,
}


def _toy_single_model(
    offsets: list[float], occupancies: list[float], altlocs: list[str]
) -> gemmi.Model:
    """Build one model holding a toy residue (a glycine), optionally with altloc conformers.

    The shared atoms (N, the removable H, CA) are written once, blank altloc and full
    occupancy. Each conformer adds its own copy of the conformer atoms (C, O) under its
    altloc ID, so several conformers give one residue with altlocs (e.g. A at 0.4 and B at
    0.6), while one conformer with a blank altloc gives a plain residue.

    Parameters
    ----------
    offsets
        x-axis shift of each conformer's C and O.
    occupancies
        Occupancy of each conformer.
    altlocs
        Altloc ID of each conformer, e.g. ``["A", "B"]``, or ``[" "]`` for a plain residue.

    Returns
    -------
    gemmi.Model
        One unnamed model with chain A holding the residue.
    """
    residue = gemmi.Residue()
    residue.name = "GLY"
    residue.seqid = gemmi.SeqId(1, " ")
    residue.het_flag = "A"

    def add_atom(name: str, element: str, x: float, occupancy: float, altloc: str) -> None:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x, 2.0, 3.0)
        atom.b_iso = 15.0
        atom.occ = occupancy
        atom.altloc = altloc
        residue.add_atom(atom)

    for name, element, x in (("N", "N", 1.0), ("H", "H", 1.5), ("CA", "C", 2.0)):
        add_atom(name, element, x, 1.0, BLANK_ALTLOC_ID)
    for x_offset, occupancy, altloc in zip(offsets, occupancies, altlocs, strict=True):
        for name, element, x in (("C", "C", 3.0), ("O", "O", 4.0)):
            add_atom(name, element, x + x_offset, occupancy, altloc)

    chain = gemmi.Chain("A")
    chain.add_residue(residue)
    model = gemmi.Model("")
    model.add_chain(chain)
    return model


def _write_models(path: Path, models: list[gemmi.Model]) -> None:
    """Number the models 1..N and write them as one crystallographic PDB file.

    Parameters
    ----------
    path
        Output PDB path.
    models
        Models to write, in order.

    Notes
    -----
    This function writes a PDB file at ``path`` and renames each model in ``models``.
    """
    structure = gemmi.Structure()
    for model_index, model in enumerate(models, start=1):
        model.name = str(model_index)
        structure.add_model(model)
    structure.cell = gemmi.UnitCell(*CELL)
    structure.spacegroup_hm = SPACEGROUP
    structure.write_pdb(str(path))


@pytest.fixture
def multi_model_path(tmp_path: Path) -> Path:
    """Write the toy ensemble as two blank-altloc models at occupancy 1.0, one per offset.

    Parameters
    ----------
    tmp_path
        Temporary directory for the ensemble.

    Returns
    -------
    Path
        Path to the written ``multi_model.pdb``.
    """
    path = tmp_path / "multi_model.pdb"
    _write_models(
        path, [_toy_single_model([offset], [1.0], [BLANK_ALTLOC_ID]) for offset in OFFSETS]
    )
    return path


@pytest.fixture
def single_model_path(tmp_path: Path) -> Path:
    """Write the toy glycine as a single-model structure without altlocs.

    Parameters
    ----------
    tmp_path
        Temporary directory for the structure.

    Returns
    -------
    Path
        Path to the written ``single_model.pdb``.
    """
    path = tmp_path / "single_model.pdb"
    _write_models(path, [_toy_single_model([0.0], [1.0], [BLANK_ALTLOC_ID])])
    return path


class TestEnsembleSum:
    """The per-model Fprotein sum and the bulk-solvent modes built on it."""

    def test_summed_models_match_merged_altloc_structure(
        self, tmp_path: Path, multi_model_path: Path
    ) -> None:
        """Verify the per-model sum reproduces the merged altloc structure's MTZ.

        This is the assumption the script rests on: Fprotein is linear in occupancy-weighted
        atoms, so splitting altlocs into models and summing loses nothing.

        Parameters
        ----------
        tmp_path
            Temporary directory for structures and generated outputs.
        multi_model_path
            Two-model toy ensemble.
        """
        # Same populations two ways: on the altloc atoms here, per model via occupancy_values.
        _write_models(
            tmp_path / "multi_altlocs.pdb", [_toy_single_model(OFFSETS, OCCUPANCIES, ALTLOC_IDS)]
        )

        _process_single_row(
            row=BatchRowForMTZ(filename="multi_altlocs.pdb", mtzfile="multi_altlocs.mtz"),
            base_dir=tmp_path,
            output_dir=tmp_path,
            occupancy_mode="default",
            **SHARED_SETTINGS,
        )
        _process_single_row(
            row=BatchRowForMTZ(
                filename=multi_model_path.name,
                occupancy_values=OCCUPANCIES,
                mtzfile="multi_model.mtz",
            ),
            base_dir=tmp_path,
            output_dir=tmp_path,
            **SHARED_SETTINGS,
        )

        multi_altlocs = rs.read_mtz(str(tmp_path / "multi_altlocs.mtz")).sort_index()
        multi_model = rs.read_mtz(str(tmp_path / "multi_model.mtz")).sort_index()
        assert list(multi_model.columns) == list(multi_altlocs.columns)
        assert multi_model.index.equals(multi_altlocs.index)
        assert multi_model.cell.parameters == multi_altlocs.cell.parameters
        assert multi_model.spacegroup.hm == multi_altlocs.spacegroup.hm
        # R-free flags depend only on the reflection list and seed, so they must match exactly.
        flag_columns = multi_altlocs.columns.difference(["Fprotein", "SIGFprotein", "PHIFprotein"])
        assert multi_model[flag_columns].equals(multi_altlocs[flag_columns])
        # Compare complex values so phases of near-zero amplitudes cannot fail on wrap-around.
        multi_altlocs_f = multi_altlocs.to_structurefactor("Fprotein", "PHIFprotein").to_numpy()
        multi_model_f = multi_model.to_structurefactor("Fprotein", "PHIFprotein").to_numpy()
        np.testing.assert_allclose(
            multi_model_f, multi_altlocs_f, atol=1e-4 * np.abs(multi_altlocs_f).max()
        )

    @pytest.mark.parametrize("bulk_solvent", ["combined", "per_conformer"])
    def test_identical_models_match_single_structure_with_solvent(
        self, tmp_path: Path, single_model_path: Path, bulk_solvent: str
    ) -> None:
        """Verify both solvent modes reproduce the single-structure Ftotal for a degenerate
        ensemble.

        Two identical models at populations 0.4/0.6 are one structure, so the combined mask,
        the population-weighted per-model masks, and the single-structure mask must agree.

        Parameters
        ----------
        tmp_path
            Temporary directory for structures and generated outputs.
        single_model_path
            Single-model toy structure.
        bulk_solvent
            Bulk-solvent mode under test.
        """
        _write_models(
            tmp_path / "identical_models.pdb",
            [_toy_single_model([offset], [1.0], [BLANK_ALTLOC_ID]) for offset in [0.0, 0.0]],
        )

        _process_single_row(
            row=BatchRowForMTZ(filename=single_model_path.name, mtzfile="single_model.mtz"),
            base_dir=tmp_path,
            output_dir=tmp_path,
            occupancy_mode="default",
            bulk_solvent="combined",
            **SHARED_SETTINGS,
        )
        _process_single_row(
            row=BatchRowForMTZ(
                filename="identical_models.pdb",
                occupancy_values=OCCUPANCIES,
                mtzfile="identical_models.mtz",
            ),
            base_dir=tmp_path,
            output_dir=tmp_path,
            bulk_solvent=bulk_solvent,
            **SHARED_SETTINGS,
        )

        single = rs.read_mtz(str(tmp_path / "single_model.mtz")).sort_index()
        multi_model = rs.read_mtz(str(tmp_path / "identical_models.mtz")).sort_index()
        assert list(multi_model.columns) == list(single.columns)
        assert multi_model.index.equals(single.index)
        for label in ("protein", "total"):
            # Compare complex values so phases of near-zero amplitudes cannot fail on wrap-around.
            single_f = single.to_structurefactor(f"F{label}", f"PHIF{label}").to_numpy()
            multi_model_f = multi_model.to_structurefactor(f"F{label}", f"PHIF{label}").to_numpy()
            np.testing.assert_allclose(multi_model_f, single_f, atol=1e-4 * np.abs(single_f).max())

    def test_solvent_modes_leave_fprotein_unchanged(self, multi_model_path: Path) -> None:
        """Verify bulk solvent only adds an Ftotal set beside an unchanged Fprotein set.

        Parameters
        ----------
        multi_model_path
            Two-model toy ensemble; its directory also receives the generated outputs.
        """
        output_dir = multi_model_path.parent
        datasets = {}
        for bulk_solvent in ("off", "combined", "per_conformer"):
            _process_single_row(
                row=BatchRowForMTZ(
                    filename=multi_model_path.name,
                    occupancy_values=OCCUPANCIES,
                    mtzfile=f"{bulk_solvent}.mtz",
                ),
                base_dir=output_dir,
                output_dir=output_dir,
                bulk_solvent=bulk_solvent,
                **SHARED_SETTINGS,
            )
            mtz_path = output_dir / f"{bulk_solvent}.mtz"
            datasets[bulk_solvent] = rs.read_mtz(str(mtz_path)).sort_index()

        assert "Ftotal" not in datasets["off"].columns
        for bulk_solvent in ("combined", "per_conformer"):
            dataset = datasets[bulk_solvent]
            assert {"Ftotal", "SIGFtotal", "PHIFtotal"} <= set(dataset.columns)
            assert np.isfinite(dataset["Ftotal"].to_numpy()).all()
            np.testing.assert_array_equal(
                dataset.to_structurefactor("Fprotein", "PHIFprotein").to_numpy(),
                datasets["off"].to_structurefactor("Fprotein", "PHIFprotein").to_numpy(),
            )


class TestPopulations:
    """How ``load_models`` turns occupancy options into per-model populations."""

    def test_multi_model_populations_use_shared_occupancy_policy(
        self, multi_model_path: Path
    ) -> None:
        """Verify model populations are resolved like altloc occupancies: trimmed or zero-padded
        according to the number of supplied values and the number of conformers.

        The policy itself is covered by ``TestAssignOccupancies`` in ``test_synthetic_utils``;
        this only checks the multi-model path goes through it rather than refusing a mismatch.

        Parameters
        ----------
        multi_model_path
            Two-model toy ensemble.
        """
        _, trimmed = load_models(multi_model_path, "custom", [0.4, 0.6, 0.0])
        _, padded = load_models(multi_model_path, "custom", [1.0])

        assert trimmed == [0.4, 0.6]
        assert padded == [1.0, 0.0]

    def test_multi_model_ignores_file_occupancy_and_altloc_label(
        self, tmp_path: Path, multi_model_path: Path
    ) -> None:
        """Verify a multi-model file is weighted only by populations (occupancy per conformer)
        rather than by atomic occupancies.

        Each model assumes atom occupancy of 1.0 regardless of the occupancy field of that atom
        from the input file. The non-blank altloc ID is also ignored since it is meant to be a
        label for eval scripts rather than actually indicative of altlocs within a model.

        Parameters
        ----------
        tmp_path
            Temporary directory for the labeled ensemble and generated outputs.
        multi_model_path
            Two-model toy ensemble with blank altlocs at occupancy 1.0.
        """
        labeled_path = tmp_path / "labeled.pdb"
        _write_models(
            labeled_path,
            [
                _toy_single_model([offset], [occupancy], [altloc])
                for offset, occupancy, altloc in zip(OFFSETS, OCCUPANCIES, ALTLOC_IDS, strict=True)
            ],
        )

        for path in (multi_model_path, labeled_path):
            _process_single_row(
                row=BatchRowForMTZ(
                    filename=path.name, occupancy_values=OCCUPANCIES, mtzfile=f"{path.stem}.mtz"
                ),
                base_dir=tmp_path,
                output_dir=tmp_path,
                **SHARED_SETTINGS,
            )

        labeled_mtz = (tmp_path / "labeled.mtz").read_bytes()
        assert labeled_mtz == (tmp_path / "multi_model.mtz").read_bytes()

    def test_uniform_mode_needs_no_occupancies_for_multi_model(
        self, multi_model_path: Path
    ) -> None:
        """Verify uniform mode gives every model of a multi-model file an equal population.

        Parameters
        ----------
        multi_model_path
            Two-model toy ensemble.
        """
        _, populations = load_models(multi_model_path, "uniform", [])

        assert populations == [0.5, 0.5]

    @pytest.mark.parametrize("occupancy_mode", ["default", "custom"])
    def test_multi_model_needs_occupancies(
        self, multi_model_path: Path, occupancy_mode: str
    ) -> None:
        """Verify a multi-model file without occupancies is refused unless uniform is asked for.

        Parameters
        ----------
        multi_model_path
            Two-model toy ensemble.
        occupancy_mode
            Occupancy mode that cannot supply per-model occupancies by itself.
        """
        with pytest.raises(ValueError, match="one population per model"):
            load_models(multi_model_path, occupancy_mode, [])

    def test_single_model_keeps_altloc_occupancies(self, tmp_path: Path) -> None:
        """Verify a single model is one conformer whose occupancy options apply to its altlocs.

        Parameters
        ----------
        tmp_path
            Temporary directory for the structure.
        """
        _write_models(
            tmp_path / "multi_altlocs.pdb", [_toy_single_model(OFFSETS, OCCUPANCIES, ALTLOC_IDS)]
        )

        models, populations = load_models(tmp_path / "multi_altlocs.pdb", "custom", [0.3, 0.7])

        assert populations == [1.0]
        (atom_array,) = models
        for altloc, occupancy in zip(ALTLOC_IDS, [0.3, 0.7], strict=True):
            np.testing.assert_allclose(
                atom_array.occupancy[atom_array.altloc_id == altloc], occupancy
            )

    def test_model_with_several_altloc_ids_is_refused(self, tmp_path: Path) -> None:
        """Verify a model of a multi-model file must be a single conformer.

        Parameters
        ----------
        tmp_path
            Temporary directory for the ensemble.
        """
        _write_models(
            tmp_path / "multi_model.pdb",
            [_toy_single_model(OFFSETS, OCCUPANCIES, ALTLOC_IDS) for _ in range(2)],
        )

        with pytest.raises(ValueError, match="at most one non-blank altloc ID"):
            load_models(tmp_path / "multi_model.pdb", "custom", OCCUPANCIES)


class TestRowsAndBatch:
    """Per-row inputs and outputs, and batch processing over a CSV."""

    def test_batch_csv_matches_single_ensemble_mode(
        self, tmp_path: Path, multi_model_path: Path
    ) -> None:
        """Verify a CSV row naming a multi-model file writes the single-mode MTZ.

        Parameters
        ----------
        tmp_path
            Temporary directory for the CSV and generated outputs.
        multi_model_path
            Two-model toy ensemble.
        """
        csv_path = tmp_path / "ensembles.csv"
        csv_path.write_text(
            f"filename,occupancy_values,mtzfile\n{multi_model_path.name},0.4:0.6,ab.mtz\n"
        )
        batch_dir = tmp_path / "batch"

        process_batch(
            csv_path=csv_path, base_dir=tmp_path, output_dir=batch_dir, n_jobs=1, **SHARED_SETTINGS
        )
        _process_single_row(
            row=BatchRowForMTZ(
                filename=multi_model_path.name, occupancy_values=OCCUPANCIES, mtzfile="ab.mtz"
            ),
            base_dir=tmp_path,
            output_dir=tmp_path,
            **SHARED_SETTINGS,
        )

        assert (batch_dir / "ab.mtz").read_bytes() == (tmp_path / "ab.mtz").read_bytes()

    def test_saved_structure_keeps_one_model_per_conformer(
        self, tmp_path: Path, multi_model_path: Path
    ) -> None:
        """Verify the saved SF input holds every processed model at occupancy 1.0.

        Parameters
        ----------
        tmp_path
            Temporary directory for generated outputs.
        multi_model_path
            Two-model toy ensemble.
        """
        _process_single_row(
            row=BatchRowForMTZ(
                filename=multi_model_path.name,
                occupancy_values=OCCUPANCIES,
                mtzfile="out/multi_model.mtz",
            ),
            base_dir=tmp_path,
            output_dir=tmp_path,
            save_structure=True,
            **SHARED_SETTINGS,
        )

        saved_path = tmp_path / "out" / "multi_model_sf_input.cif"
        assert len(gemmi.read_structure(str(saved_path))) == len(OFFSETS)
        for model, offset in enumerate(OFFSETS):
            saved = load_structure_with_altlocs(saved_path, model=model)
            assert list(saved.atom_name) == ["N", "CA", "C", "O"]  # hydrogen stripped
            # Only the conformer atoms C and O are shifted; N and CA are shared.
            np.testing.assert_allclose(saved.coord[:, 0], [1.0, 2.0, 3.0 + offset, 4.0 + offset])
            np.testing.assert_array_equal(saved.occupancy, 1.0)
            np.testing.assert_array_equal(saved.b_factor, 20.0)

    @pytest.mark.parametrize(
        ("unit_cell", "space_group"),
        [
            ((40.0, 41.0, 42.0, 90.0, 90.0, 90.0), "1"),
            ((35.0, 36.0, 37.0, 90.0, 90.0, 90.0), "16"),
        ],
    )
    def test_row_unit_cell_and_space_group_override_file(
        self, single_model_path: Path, unit_cell: tuple[float, ...], space_group: str
    ) -> None:
        """Verify per-row unit cell and space group replace the ones in the structure file.

        Parameters
        ----------
        single_model_path
            Single-model toy structure.
        unit_cell
            Unit cell parameters written to the row.
        space_group
            Space group number written to the row.
        """
        row = BatchRowForMTZ.from_dict(
            {
                "filename": single_model_path.name,
                "mtzfile": "override.mtz",
                "unit_cell": ":".join(f"{value:g}" for value in unit_cell),
                "space_group": space_group,
            }
        )
        _process_single_row(
            row=row,
            base_dir=single_model_path.parent,
            output_dir=single_model_path.parent,
            **SHARED_SETTINGS,
        )
        dataset = rs.read_mtz(str(single_model_path.parent / "override.mtz"))

        assert dataset.cell.parameters == pytest.approx(unit_cell)
        assert dataset.spacegroup.hm == gemmi.SpaceGroup(int(space_group)).hm

    def test_failed_row_does_not_stop_the_batch(
        self, tmp_path: Path, single_model_path: Path
    ) -> None:
        """Verify a row that fails to load is skipped and later rows still write their MTZ.

        Parameters
        ----------
        tmp_path
            Temporary directory for the broken structure, the CSV, and generated outputs.
        single_model_path
            Single-model toy structure for the row that should succeed.
        """
        (tmp_path / "broken.pdb").write_text("not a structure\n")
        csv_path = tmp_path / "rows.csv"
        csv_path.write_text(
            f"filename,mtzfile\nbroken.pdb,broken.mtz\n{single_model_path.name},single_model.mtz\n"
        )
        batch_dir = tmp_path / "batch"

        process_batch(
            csv_path=csv_path, base_dir=tmp_path, output_dir=batch_dir, n_jobs=1, **SHARED_SETTINGS
        )

        assert not (batch_dir / "broken.mtz").exists()
        assert (batch_dir / "single_model.mtz").exists()
