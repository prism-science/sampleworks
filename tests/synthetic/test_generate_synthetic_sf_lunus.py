"""Tests for the lunus structure-factor generator.

Self-consistency tests check properties that follow from the definitions: an
ensemble of identical copies has zero diffuse intensity, and in P1 the diffuse
term is invariant to a rigid translation. The cross-engine test checks ``<F>``
against SFcalculator. The MTZ writer tests check the column types written.

Thresholds come from values measured on 6B8X (P 31 2 1, 3011 polymer atoms) at
1.8 A on CPU, recorded in each test; the cross-engine test runs at 2.2 A because
SFcalculator's memory scales with the reflection count.
"""

from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArrayStack
from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi
from sampleworks.utils.atom_array_utils import keep_amino_acids, keep_polymer


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Below the guard: the generator imports lunus.sf at module scope.
from sampleworks.core.forward_models.xray import lunus_sf
from sampleworks.synthetic.generate_synthetic_sf_lunus import (
    compute_ensemble_amplitudes,
    dataset_from_bragg_amplitudes,
    dataset_from_diffuse_intensities,
    save_mtz,
)


RESOLUTION = 1.8

# SFcalculator builds [n_atoms, n_refl] tensors, so its memory is linear in the
# reflection count: 25514 reflections at 2.2 A took 4.94 GB peak RSS on 6B8X.
CROSS_ENGINE_RESOLUTION = 2.2

SOURCE_PDB = "6b8x_final.pdb"


@pytest.fixture(scope="module")
def cpu_device() -> torch.device:
    """CPU, where the tolerances were measured; the session ``device`` fixture may be a GPU."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def polymer_6b8x(structure_6b8x_with_altlocs):
    """6B8X's polymer from the session fixture, waters and glycerol removed.

    6B8X has B_min 2.0 with two thirds of its atoms under B 10, so it exercises
    the grid-sampling regime :func:`recommended_blur` is for.
    """
    atom_array = structure_6b8x_with_altlocs
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    return keep_polymer(keep_amino_acids(remove_waters(atom_array)))


@pytest.fixture(scope="module")
def crystal_6b8x(resources_dir: Path):
    """Unit cell and space group read from the deposited file."""
    meta = gemmi.read_structure(str(resources_dir / "6b8x" / SOURCE_PDB))
    return meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)


def _amplitudes(atom_array, coords, cell, spacegroup, device, *, resolution=RESOLUTION, **kwargs):
    """Run the lunus generator's compute step and return (hkl, <F>, diffuse)."""
    return compute_ensemble_amplitudes(
        atom_array, coords, cell, spacegroup, resolution, device, **kwargs
    )


class TestSelfConsistency:
    """Properties that follow from the definitions, independent of any other engine."""

    def test_identical_configurations_have_zero_diffuse(
        self, polymer_6b8x, crystal_6b8x, cpu_device
    ):
        """N copies of one structure have <|F|^2> == |<F>|^2, so diffuse is zero."""
        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x
        replicated = np.repeat(atom_array.coord[None], 4, axis=0)

        _, mean_f, diffuse = _amplitudes(atom_array, replicated, cell, spacegroup, cpu_device)

        # The residual is float32 cancellation between <|F|^2> and |<F>|^2, which
        # scales with the intensity, so it is normalized RMS to RMS. Measured on
        # 6B8X at 1.8 A: 6e-9 to 4.3e-8 across runs (the grid reduction order varies).
        intensity = np.abs(mean_f).astype(np.float64) ** 2
        rms_intensity = float(np.sqrt(np.mean(intensity**2)))
        rms_diffuse = float(np.sqrt(np.mean(diffuse.astype(np.float64) ** 2)))
        assert rms_intensity > 0, "degenerate <F>; the calculation produced nothing"

        normalized_residual = rms_diffuse / rms_intensity
        print(f"\nidentical-ensemble diffuse residual, normalized: {normalized_residual:.2e}")
        assert normalized_residual < 1e-6

    def test_single_configuration_matches_its_own_replication(
        self, polymer_6b8x, crystal_6b8x, cpu_device
    ):
        """<F> over N identical copies equals F of one copy."""
        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x

        _, single, _ = _amplitudes(atom_array, atom_array.coord[None], cell, spacegroup, cpu_device)
        _, replicated, _ = _amplitudes(
            atom_array, np.repeat(atom_array.coord[None], 3, axis=0), cell, spacegroup, cpu_device
        )

        # A norm ratio, since float32 accumulation order puts a few weak reflections
        # past any fixed atol. Measured on 6B8X at 1.8 A: 1.4e-7 to 1.5e-7.
        deviation = float(np.linalg.norm(single - replicated) / np.linalg.norm(single))
        print(f"\nsingle vs replicated relative deviation: {deviation:.2e}")
        assert deviation < 1e-5

    @staticmethod
    def _perturbed_ensemble(coord: np.ndarray) -> np.ndarray:
        """Two different configurations, so the diffuse term is nonzero."""
        rng = np.random.default_rng(0)
        return np.stack([coord, coord + rng.normal(0, 0.3, coord.shape)])

    def test_diffuse_is_invariant_to_rigid_translation_in_p1(
        self, polymer_6b8x, crystal_6b8x, cpu_device
    ):
        """In P1, translating every configuration identically leaves diffuse unchanged.

        A common translation multiplies every F_b by exp(2 pi i h.d), so |F_b| and
        |<F>| are unchanged. The residual is grid discretization (atoms move
        relative to voxel centres), measured at 5.8e-5 on 6B8X at 1.8 A; the
        symmetry case below is 0.93.
        """
        atom_array = polymer_6b8x
        cell, _ = crystal_6b8x
        p1 = gemmi.SpaceGroup("P 1")
        ensemble = self._perturbed_ensemble(atom_array.coord)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, p1, cpu_device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, p1, cpu_device
        )

        assert float(np.mean(diffuse)) > 0, "no diffuse signal to test invariance of"
        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\nP1 diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation < 1e-2

    def test_diffuse_is_not_invariant_under_symmetry(self, polymer_6b8x, crystal_6b8x, cpu_device):
        """Translating the ASU in a non-P1 group changes diffuse.

        In F_total(h) = sum_g exp(2 pi i h.t_g) F_ASU(h R_g), a translation d
        multiplies each term by exp(2 pi i (h R_g).d), which depends on g and does
        not factor out. So diffuse scoring is sensitive to the model's absolute
        position in a crystal. Measured on 6B8X (P 31 2 1) at 1.8 A: relative
        deviation 0.93. If this passes, symmetry expansion has stopped happening.
        """
        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x
        assert spacegroup.hm != "P 1", "this test needs a non-trivial space group"
        ensemble = self._perturbed_ensemble(atom_array.coord)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, spacegroup, cpu_device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, spacegroup, cpu_device
        )

        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\n{spacegroup.hm} diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation > 0.1


class TestCrossEngineAgreement:
    """lunus versus SFcalculator on the same structure."""

    # Measured on 6B8X at 2.2 A, 25514 reflections: correlation 1.000000, R 0.0001,
    # scale 1.0000. With blur 0 the same comparison gives correlation 0.991505 and
    # R 0.1007 at full reflection coverage, which is why build_setup applies
    # recommended_blur. Remeasure if CROSS_ENGINE_RESOLUTION changes.
    MIN_CORRELATION = 0.9999
    MAX_R_FACTOR = 0.009

    # Both coverages measured 1.0000 at 3.0, 2.5 and 2.2 A. A low value means the
    # two ASU conventions disagree.
    MIN_COVERAGE = 0.99

    @pytest.fixture(scope="class")
    @staticmethod
    def sfcalculator_amplitudes(polymer_6b8x, crystal_6b8x, cpu_device):
        """|F| from SFcalculator on the same atoms, indexed by Miller index."""
        # Inline so the import follows the importorskip guard.
        pytest.importorskip("SFC_Torch", reason="sfcalculator-torch not installed")
        from SFC_Torch import SFcalculator
        from SFC_Torch.io import PDBParser

        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x
        gemmi_structure = atomarray_to_gemmi(atom_array, cell, spacegroup.hm)

        sfc = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=CROSS_ENGINE_RESOLUTION,
            mode="xray",
            anomalous=False,
            set_experiment=False,
            device=cpu_device,
        )
        sfc.calc_fprotein()
        hkl = np.asarray(sfc.Hasu_array, dtype=np.int64)
        amplitude = torch.abs(sfc.Fprotein_asu).detach().cpu().numpy()
        return {tuple(h): a for h, a in zip(hkl, amplitude, strict=True)}

    def test_amplitudes_agree_with_sfcalculator(
        self, polymer_6b8x, crystal_6b8x, cpu_device, sfcalculator_amplitudes
    ):
        """Correlation and R-factor over the reflections both engines produced.

        The reflection sets are generated independently (gemmi's ASU here,
        SFcalculator's own there), so the comparison is over their intersection.
        """
        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x

        hkl, mean_f, _ = _amplitudes(
            atom_array,
            atom_array.coord[None],
            cell,
            spacegroup,
            cpu_device,
            resolution=CROSS_ENGINE_RESOLUTION,
        )
        lunus_amplitude = np.abs(mean_f)

        shared = [
            (a, sfcalculator_amplitudes[tuple(h)])
            for h, a in zip(hkl, lunus_amplitude, strict=True)
            if tuple(h) in sfcalculator_amplitudes
        ]
        # Coverage both ways: a truncated shell or mis-sized grid drops reflections
        # without perturbing the survivors, so one-sided coverage would pass.
        lunus_coverage = len(shared) / len(hkl)
        sfc_coverage = len(shared) / len(sfcalculator_amplitudes)
        print(
            f"\nreflection-set overlap: {len(shared)} shared, "
            f"{lunus_coverage:.4f} of lunus's {len(hkl)}, "
            f"{sfc_coverage:.4f} of SFcalculator's {len(sfcalculator_amplitudes)}"
        )
        assert min(lunus_coverage, sfc_coverage) > self.MIN_COVERAGE, (
            f"{len(shared)} reflections shared: {lunus_coverage:.4f} of lunus's "
            f"{len(hkl)} and {sfc_coverage:.4f} of SFcalculator's "
            f"{len(sfcalculator_amplitudes)}; the two ASU conventions disagree"
        )

        lunus_shared = np.array([s[0] for s in shared])
        sfc_shared = np.array([s[1] for s in shared])

        # Scale-invariant R-factor.
        scale = float(np.sum(lunus_shared * sfc_shared) / np.sum(lunus_shared**2))
        correlation = float(np.corrcoef(lunus_shared, sfc_shared)[0, 1])
        r_factor = float(
            np.sum(np.abs(scale * lunus_shared - sfc_shared)) / np.sum(np.abs(sfc_shared))
        )
        print(
            f"\nlunus vs SFcalculator over {len(shared)} reflections: "
            f"correlation {correlation:.6f}, R {r_factor:.4f}, scale {scale:.4f}"
        )

        assert correlation > self.MIN_CORRELATION
        assert r_factor < self.MAX_R_FACTOR


class TestMTZWriters:
    """Round trips through the two MTZ writers, the module's only reciprocalspaceship code.

    These pin the MTZ column types, which rs infers from column names.
    """

    @staticmethod
    def reflections(n: int = 64) -> np.ndarray:
        """A small block of Miller indices, origin excluded."""
        h, k, l = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1).astype(np.int32)
        return hkl[1 : n + 1]

    def test_amplitudes_round_trip_through_mtz(self, tmp_path):
        """Amplitude, sigma and phase come back as MTZ F/Q/P with the cell and space group."""
        hkl = self.reflections()
        structure_factors = (np.arange(1, len(hkl) + 1) * (1.0 + 0.5j)).astype(np.complex64)
        cell = gemmi.UnitCell(31.7, 42.3, 55.9, 90.0, 104.5, 90.0)
        path = tmp_path / "amplitudes.mtz"

        save_mtz(
            dataset_from_bragg_amplitudes(
                hkl,
                structure_factors,
                cell,
                gemmi.SpaceGroup("P 1 21 1"),
                label="MODEL",
                sigma_f_scale=0.2,
                test_fraction=0.0,
            ),
            path,
            "structure factors",
        )

        mtz = gemmi.read_mtz_file(str(path))
        assert [(c.label, c.type) for c in mtz.columns] == [
            ("H", "H"),
            ("K", "H"),
            ("L", "H"),
            ("FMODEL", "F"),
            ("SIGFMODEL", "Q"),
            ("PHIFMODEL", "P"),
        ]
        assert mtz.spacegroup.hm == "P 1 21 1"
        assert mtz.cell.a == pytest.approx(31.7, abs=1e-3)

        amplitude = np.abs(structure_factors)
        np.testing.assert_allclose(mtz.column_with_label("FMODEL").array, amplitude, rtol=1e-6)
        np.testing.assert_allclose(
            mtz.column_with_label("SIGFMODEL").array, amplitude * 0.2, rtol=1e-6
        )
        np.testing.assert_allclose(
            mtz.column_with_label("PHIFMODEL").array,
            np.rad2deg(np.angle(structure_factors)),
            rtol=1e-5,
        )

    def test_intensities_round_trip_through_mtz(self, tmp_path):
        """Diffuse intensities come back as MTZ type J, the type lunus xtraj writes."""
        hkl = self.reflections()
        intensities = np.linspace(-0.5, 10.0, len(hkl)).astype(np.float32)
        path = tmp_path / "diffuse.mtz"

        save_mtz(
            dataset_from_diffuse_intensities(
                hkl,
                intensities,
                gemmi.UnitCell(31.7, 42.3, 55.9, 90.0, 104.5, 90.0),
                gemmi.SpaceGroup("P 1 21 1"),
            ),
            path,
            "diffuse intensities",
        )

        mtz = gemmi.read_mtz_file(str(path))
        assert [(c.label, c.type) for c in mtz.columns] == [
            ("H", "H"),
            ("K", "H"),
            ("L", "H"),
            ("ID", "J"),
        ]
        # Slightly negative diffuse values are written as computed, not clipped.
        np.testing.assert_allclose(mtz.column_with_label("ID").array, intensities, rtol=1e-6)
        assert mtz.column_with_label("ID").array.min() < 0.0

    def test_rfree_flags_written_only_when_requested(self, tmp_path):
        """R-free flags are written only for test_fraction > 0."""
        hkl = self.reflections()
        structure_factors = np.ones(len(hkl), dtype=np.complex64)
        args = (hkl, structure_factors, gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0))

        without = tmp_path / "without.mtz"
        save_mtz(
            dataset_from_bragg_amplitudes(*args, gemmi.SpaceGroup("P 1"), test_fraction=0.0),
            without,
            "structure factors",
        )
        assert not any(c.type == "I" for c in gemmi.read_mtz_file(str(without)).columns)

        with_flags = tmp_path / "with.mtz"
        save_mtz(
            dataset_from_bragg_amplitudes(
                *args, gemmi.SpaceGroup("P 1"), test_fraction=0.25, seed=7
            ),
            with_flags,
            "structure factors",
        )
        flags = gemmi.read_mtz_file(str(with_flags)).column_with_label("R-free-flags")
        assert flags.type == "I"
        assert set(np.unique(flags.array)) <= {0.0, 1.0}
        assert 0.0 < flags.array.mean() < 1.0

    def test_intensity_type_survives_an_unconventional_label(self, tmp_path):
        """The intensity type is J for any label; rs alone would infer R for this one."""
        hkl = self.reflections(8)
        path = tmp_path / "labelled.mtz"

        save_mtz(
            dataset_from_diffuse_intensities(
                hkl,
                np.ones(len(hkl), dtype=np.float32),
                gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0),
                gemmi.SpaceGroup("P 1"),
                label="DIFFUSE",
            ),
            path,
            "diffuse intensities",
        )

        column = gemmi.read_mtz_file(str(path)).column_with_label("DIFFUSE")
        assert column.type == "J"

    def test_builders_return_indexed_datasets_with_mtz_dtypes(self, tmp_path):
        """The builders write nothing and return datasets with MTZ dtypes."""
        hkl = self.reflections(8)
        cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
        space_group = gemmi.SpaceGroup("P 1")

        amplitudes = dataset_from_bragg_amplitudes(
            hkl, np.ones(len(hkl), dtype=np.complex64), cell, space_group, test_fraction=0.0
        )
        intensities = dataset_from_diffuse_intensities(
            hkl, np.ones(len(hkl), dtype=np.float32), cell, space_group
        )

        assert list(tmp_path.iterdir()) == []
        assert amplitudes.index.names == ["H", "K", "L"]
        assert [str(dtype) for dtype in amplitudes.dtypes] == ["SFAmplitude", "Stddev", "Phase"]
        assert [str(dtype) for dtype in intensities.dtypes] == ["Intensity"]


class TestSolvent:
    """The bulk-solvent mask, calibrated against gemmi's geometric mask by default."""

    def test_calibrated_cutoff_reproduces_the_geometric_fraction(
        self, polymer_6b8x, crystal_6b8x, cpu_device
    ):
        from lunus.sf import mask_occupancy, splat_density, symmetrize_sum

        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x
        setup = lunus_sf.build_setup(atom_array, cell, spacegroup.hm, 2.2, device=cpu_device)
        target = lunus_sf.geometric_solvent_fraction(
            atom_array, atom_array.coord, cell, spacegroup, setup.grid_shape
        )
        coords = torch.as_tensor(atom_array.coord, dtype=torch.float32)
        occupancies = torch.as_tensor(np.asarray(atom_array.occupancy, dtype=np.float32))

        solvent = lunus_sf.calibrate_solvent(setup, coords, occupancies, target)

        density = symmetrize_sum(
            splat_density(
                coords @ setup.frac_matrix.T,
                setup.element_idx,
                occupancies,
                setup.atom_A,
                setup.atom_lam,
                setup.elem_offsets,
                setup.atom_radius_ang,
                setup.grid_shape,
                setup.orth_matrix,
                setup.taper_width,
            ),
            setup.grid_ops,
        )
        occupancy = float(
            mask_occupancy(solvent.build_mask(density, setup.orth_matrix, pipeline_blur=setup.blur))
        )
        # Measured on 6B8X: target 0.625, reproduced to 3 decimals.
        assert 0.3 < target < 0.9
        assert occupancy == pytest.approx(target, abs=0.01)
        assert solvent.taper_width == pytest.approx(0.5 * solvent.cutoff)

    def test_generator_calibrates_by_default_and_changes_the_amplitudes(
        self, polymer_6b8x, crystal_6b8x, cpu_device, caplog
    ):
        atom_array = polymer_6b8x
        cell, spacegroup = crystal_6b8x
        coords = atom_array.coord[None]

        _, without, _ = _amplitudes(
            atom_array, coords, cell, spacegroup, cpu_device, resolution=CROSS_ENGINE_RESOLUTION
        )
        _, calibrated, _ = _amplitudes(
            atom_array,
            coords,
            cell,
            spacegroup,
            cpu_device,
            resolution=CROSS_ENGINE_RESOLUTION,
            bulk_solvent=True,
        )
        _, explicit, _ = _amplitudes(
            atom_array,
            coords,
            cell,
            spacegroup,
            cpu_device,
            resolution=CROSS_ENGINE_RESOLUTION,
            bulk_solvent=True,
            solvent_cutoff=0.20,
        )

        assert "Solvent cutoff calibrated" in caplog.text
        assert not np.allclose(np.abs(calibrated), np.abs(without))
        assert not np.allclose(np.abs(calibrated), np.abs(explicit))
