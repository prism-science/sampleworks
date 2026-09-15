"""Cross-engine and self-consistency tests for the lunus structure-factor generator.

Two kinds of test live here, and they carry different weight.

**Self-consistency** (no second engine involved): an ensemble of identical copies
must have zero diffuse intensity, and the diffuse term must be invariant to a
rigid translation applied to every configuration. These pin properties that
follow from the definitions, so their thresholds are principled and they fail
loudly when the ensemble plumbing is wrong.

**Cross-engine agreement** against SFcalculator: the two engines compute the same
physics by different routes — grid splat plus FFT versus direct summation in
reciprocal space — so they agree closely but not exactly. lunus measures
correlation 0.999989 and R ≈ 0.0077 against gemmi, with its smooth density taper
the main source of the difference; SFcalculator is a third implementation, so
expect agreement of that order but not better.

That test runs at ``CROSS_ENGINE_RESOLUTION``, coarser than the rest of the
module, because SFcalculator's memory scales with the reflection count and 1.8 A
exhausts a 15 GiB machine. See the constant for the measurements.

Thresholds come from measured values, recorded in each test's docstring, per
lunus's own convention. First measured 2026-08-17 on 1VME chain A (P 1 21 1,
3357 atoms) at 1.8 A, grid 96x160x160, 86499 reflections, CPU. The cross-engine
bounds were remeasured at 2.2 A / 47499 reflections on 2026-09-06.
"""

from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Imported below the guard, not above it: the generator module imports lunus.sf
# at module scope, so without lunus this has to skip rather than fail collection.
from sampleworks.synthetic.generate_synthetic_sf_lunus import (
    compute_ensemble_amplitudes,
    dataset_from_amplitudes,
    dataset_from_intensities,
    load_configurations,
)
from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ


# Deliberately unmarked. These need neither a GPU nor model weights, and the
# whole module runs in ~7 s on CPU -- the same order as unmarked tests
# elsewhere, and as the `slow`-only precedent in
# tests/eval/test_rscc_grid_search_script.py, whose slowest test is 2.4 s.
#
# It previously carried a module-scope `slow` mark justified as "~90 s of splat
# and FFT on CPU", which does not reproduce: measured 6.8 s for all five tests
# on 2026-09-06 (4-core aarch64). Marking it hid the only coverage the diffuse
# forward model has from `pixi run -e <env> tests`, which is the loop where a
# structure-factor regression most wants to fire. Note that CI is unaffected
# either way -- `cpu-tests` deselects `gpu`, not `slow`.

RESOLUTION = 1.8

# The cross-engine test runs coarser than the rest of the module, because
# SFcalculator is the memory bound. Its F_protein builds several
# [n_atoms, n_refl] tensors before reducing over atoms (SFC_Torch/Fmodel.py),
# so peak memory is linear in the reflection count -- measured on this
# structure at 3357 atoms:
#
#     3.0 A   18858 refl   4.91 GiB
#     2.5 A   32467 refl   7.29 GiB
#     2.2 A   47499 refl   9.92 GiB
#     1.8 A   86499 refl   ~16.7 GiB (extrapolated; OOM-killed a 15 GiB machine)
#
# 2.2 A is the finest that fits with headroom. Coarser is cheaper but blunts
# the test: lunus sizes its grid from the resolution, so a coarser grid means
# more splat discretization error against SFcalculator's direct summation, and
# the R-factor bound below has to open up to match (R was 0.0009 at 2.2 A,
# 0.0026 at 2.5 A, 0.0119 at 3.0 A).
CROSS_ENGINE_RESOLUTION = 2.2

SOURCE_CIF = "1vme_final.cif"


@pytest.fixture(scope="module")
def cpu_device() -> torch.device:
    """CPU: these tests check numerics, not throughput, and must run in CI.

    Named apart from the session-wide ``device`` fixture in ``tests/conftest.py``,
    which resolves to a GPU when one is present: the tolerances below were
    measured on CPU.
    """
    return torch.device("cpu")


@pytest.fixture(scope="module")
def configurations_1vme(resources_dir: Path):
    """Chain A of 1VME with hydrogens and waters stripped, as a topology + coordinates.

    Distinct from the session-wide ``structure_1vme`` fixture, which is a parsed
    atomworks dict; this one is the ``(atom_array, coords)`` pair the lunus
    generator consumes.

    Deliberately the same selection the SFcalculator reward fixtures use
    (``tests/rewards/conftest.py``), so the two engines are compared on identical
    inputs.
    """
    source_dir = resources_dir / "1vme"
    if not (source_dir / SOURCE_CIF).exists():
        pytest.skip(f"Source structure not found at {source_dir / SOURCE_CIF}")

    row = BatchRowForMTZ(filename=SOURCE_CIF, selection="chain A")
    atom_array, coords = load_configurations(
        source_dir / SOURCE_CIF,
        row,
        occupancy_mode="default",
        strip_hydrogens=True,
        strip_waters=True,
    )
    return atom_array, coords


@pytest.fixture(scope="module")
def crystal_1vme(resources_dir: Path):
    """Unit cell and space group read from the deposited file."""
    meta = gemmi.read_structure(str(resources_dir / "1vme" / SOURCE_CIF))
    return meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)


def _amplitudes(atom_array, coords, cell, spacegroup, device, *, resolution=RESOLUTION, **kwargs):
    """Run the lunus generator's compute step and return (hkl, <F>, diffuse)."""
    return compute_ensemble_amplitudes(
        atom_array, coords, cell, spacegroup, resolution, device, **kwargs
    )


class TestSelfConsistency:
    """Properties that follow from the definitions, independent of any other engine."""

    def test_identical_configurations_have_zero_diffuse(
        self, configurations_1vme, crystal_1vme, cpu_device
    ):
        """N copies of one structure have <|F|^2> == |<F>|^2, so diffuse is zero.

        This is the sharpest check that the ensemble axis is wired correctly: if
        configurations were being summed rather than averaged, or the occupancy
        convention were doubly applied, the variance would not vanish.
        """
        atom_array, coords = configurations_1vme
        cell, spacegroup = crystal_1vme
        replicated = np.repeat(coords[:1], 4, axis=0)

        _, mean_f, diffuse = _amplitudes(atom_array, replicated, cell, spacegroup, cpu_device)

        # Diffuse is a difference of two large, nearly equal numbers, so in
        # float32 it is exactly zero only where the intensity is small. Compare
        # RMS to RMS: both are dominated by the strongest reflections, so the
        # ratio measures the relative cancellation error rather than mixing an
        # absolute residual on the largest reflection against a mean intensity.
        # Measured on 1VME chain A at 1.8 A: 3.1e-8 to 4.4e-8 across runs, i.e.
        # float32 epsilon. It varies run to run because the splat's reduction
        # order does; the bound allows for that rather than pinning one value.
        intensity = np.abs(mean_f).astype(np.float64) ** 2
        rms_intensity = float(np.sqrt(np.mean(intensity**2)))
        rms_diffuse = float(np.sqrt(np.mean(diffuse.astype(np.float64) ** 2)))
        assert rms_intensity > 0, "degenerate <F>; the calculation produced nothing"

        ratio = rms_diffuse / rms_intensity
        print(f"\nidentical-ensemble diffuse/intensity RMS ratio: {ratio:.2e}")
        assert ratio < 1e-6

    def test_single_configuration_matches_its_own_replication(
        self, configurations_1vme, crystal_1vme, cpu_device
    ):
        """<F> over N identical copies equals F of one copy.

        Guards against an ensemble weighting that scales with N.
        """
        atom_array, coords = configurations_1vme
        cell, spacegroup = crystal_1vme

        _, single, _ = _amplitudes(atom_array, coords[:1], cell, spacegroup, cpu_device)
        _, replicated, _ = _amplitudes(
            atom_array, np.repeat(coords[:1], 3, axis=0), cell, spacegroup, cpu_device
        )

        # A norm ratio rather than elementwise tolerances: the two differ only by
        # float32 accumulation order (the mean over 3 members, and a splat kernel
        # specialized to a different batch size), which shows up as a handful of
        # weak reflections exceeding any fixed atol while the fields agree to
        # ~1e-7 overall. Measured on 1VME chain A at 1.8 A: 2e-7.
        deviation = float(np.linalg.norm(single - replicated) / np.linalg.norm(single))
        print(f"\nsingle vs replicated relative deviation: {deviation:.2e}")
        assert deviation < 1e-5

    @staticmethod
    def _perturbed_ensemble(coords: np.ndarray) -> np.ndarray:
        """Two genuinely different configurations, so the diffuse term is nonzero."""
        rng = np.random.default_rng(0)
        return np.stack([coords[0], coords[0] + rng.normal(0, 0.3, coords[0].shape)])

    def test_diffuse_is_invariant_to_rigid_translation_in_p1(
        self, configurations_1vme, crystal_1vme, cpu_device
    ):
        """In P1, translating every configuration identically leaves diffuse unchanged.

        A common translation multiplies every F_b by one phase factor
        exp(2 pi i h.d), so |F_b| is unchanged and <F> changes only by that same
        phase: both <|F|^2> and |<F>|^2 are invariant.

        This holds only when no symmetry is applied -- see
        ``test_diffuse_is_not_invariant_under_symmetry`` for what happens
        otherwise, which is the case that matters for real crystals.

        Measured on 1VME chain A at 1.8 A in P1: 9.2e-4. That residual is GRID
        DISCRETIZATION, not round-off: translating the atoms moves them relative
        to voxel centres and to the tapered cutoff, so the sampled density is not
        quite the same function. Float32 noise in this pipeline is four orders
        smaller -- 3.1e-8 and 9.3e-8 in the two tests above -- and the residual
        should fall if `rate` is raised in build_setup. The bound below is an
        order of magnitude above the measurement, matching how the cross-engine
        bounds are set; what makes the result unambiguous is the contrast with
        the symmetry case, which is ~800x larger.
        """
        atom_array, coords = configurations_1vme
        cell, _ = crystal_1vme
        p1 = gemmi.SpaceGroup("P 1")
        ensemble = self._perturbed_ensemble(coords)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, p1, cpu_device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, p1, cpu_device
        )

        assert float(np.mean(diffuse)) > 0, "no diffuse signal to test invariance of"
        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\nP1 diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation < 1e-2

    def test_diffuse_is_not_invariant_under_symmetry(
        self, configurations_1vme, crystal_1vme, cpu_device
    ):
        """Translating the ASU contents in a non-P1 group DOES change diffuse.

        Characterization test for a result that corrected the plan. Translating
        the asymmetric unit by d moves a symmetry mate at Rx + t to Rx + t + Rd,
        so the packing relative to the mates changes and the crystal is a
        different structure. Equivalently, in

            F_total(h) = sum_g exp(2 pi i h.t_g) F_ASU(h R_g)

        translating by d multiplies each term by exp(2 pi i (h R_g).d), a phase
        that depends on g and so cannot factor out of the sum.

        The consequence for guidance: diffuse-only scoring is NOT blind to the
        absolute position of the model in a real crystal -- packing against
        symmetry mates makes it observable. Measured on 1VME chain A (P 1 21 1)
        at 1.8 A: 99.9% of reflections move, relative deviation ~1.0.

        If this test ever starts passing, symmetry expansion has silently stopped
        happening, which the cross-engine test would not necessarily catch.
        """
        atom_array, coords = configurations_1vme
        cell, spacegroup = crystal_1vme
        assert spacegroup.hm != "P 1", "this test needs a non-trivial space group"
        ensemble = self._perturbed_ensemble(coords)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, spacegroup, cpu_device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, spacegroup, cpu_device
        )

        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\n{spacegroup.hm} diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation > 0.1


class TestCrossEngineAgreement:
    """lunus versus SFcalculator on the same structure.

    Thresholds are provisional; see the module docstring.
    """

    # Measured on 1VME chain A at 2.2 A, 47499 reflections: correlation
    # 0.999999, R 0.0009, scale 1.0000 -- better than lunus's own agreement with
    # gemmi (0.999989 / 0.0077), the two engines' shared IT92 coefficients and
    # identical atom input leaving little room to disagree. Bounds are set an
    # order of magnitude looser than measured, to tolerate platform variation
    # without admitting a real regression.
    #
    # Both improve monotonically with resolution (R 0.0119 / 0.0026 / 0.0009 at
    # 3.0 / 2.5 / 2.2 A, extrapolating onto the 0.0002 recorded at 1.8 A), so
    # these bounds are tied to CROSS_ENGINE_RESOLUTION and must be remeasured
    # if it moves.
    MIN_CORRELATION = 0.9999
    MAX_R_FACTOR = 0.009

    # Unlike the two above, this is NOT a measured tolerance -- it is a floor
    # that detects the engines disagreeing about which reflections are in the
    # ASU at all. Both coverages measured exactly 1.0000 at 3.0, 2.5 and 2.2 A,
    # so this sits just under a complete intersection rather than at the loose
    # placeholder it started as.
    MIN_COVERAGE = 0.99

    @pytest.fixture(scope="class")
    @staticmethod
    def sfcalculator_amplitudes(configurations_1vme, crystal_1vme, cpu_device):
        """|F| from SFcalculator on the same atoms, indexed by Miller index."""
        pytest.importorskip("SFC_Torch", reason="sfcalculator-torch not installed")
        from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi
        from SFC_Torch import SFcalculator
        from SFC_Torch.io import PDBParser

        atom_array, _ = configurations_1vme
        cell, spacegroup = crystal_1vme
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
        self, configurations_1vme, crystal_1vme, cpu_device, sfcalculator_amplitudes
    ):
        """Correlation and R-factor over the reflections both engines produced.

        The two reflection sets are generated independently (gemmi's ASU here,
        SFcalculator's own there), so the comparison is over their intersection.
        A small intersection is itself a failure -- it would mean the ASU
        conventions disagree.
        """
        atom_array, coords = configurations_1vme
        cell, spacegroup = crystal_1vme

        hkl, mean_f, _ = _amplitudes(
            atom_array,
            coords[:1],
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
        # Coverage BOTH ways. A one-sided check passes when lunus emits a small
        # subset of SFcalculator's reflections, which is a real failure mode: a
        # truncated resolution shell or a mis-sized grid drops reflections
        # without perturbing the ones that survive, so the amplitude agreement
        # below would still look perfect over a shrunken intersection.
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

        # Scale-invariant: the engines share a convention in principle, but a
        # constant factor is not what this test is for.
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
    """Round-trip coverage for the two MTZ writers.

    These are the module's only rs-dependent code, so they are also the test
    Marcus asked for in PR #380: run over the writers and a lost
    reciprocalspaceship dependency fails here rather than silently downstream.

    What they pin is the MTZ column *types*, the letters downstream tools
    dispatch on. rs assigns them by inferring from the column names, so a
    renamed column would silently change a type without these.
    """

    @staticmethod
    def reflections(n: int = 64) -> np.ndarray:
        """A small block of Miller indices, origin excluded."""
        h, k, l = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1).astype(np.int32)
        return hkl[1 : n + 1]

    def test_amplitudes_round_trip_through_mtz(self, tmp_path):
        """Amplitude, sigma and phase must come back as MTZ F/Q/P with their
        values intact, and the crystal metadata must survive the write."""
        hkl = self.reflections()
        structure_factors = (np.arange(1, len(hkl) + 1) * (1.0 + 0.5j)).astype(np.complex64)
        cell = gemmi.UnitCell(31.7, 42.3, 55.9, 90.0, 104.5, 90.0)
        path = tmp_path / "amplitudes.mtz"

        dataset_from_amplitudes(
            hkl,
            structure_factors,
            cell,
            gemmi.SpaceGroup("P 1 21 1"),
            label="MODEL",
            sigma_f_scale=0.2,
            test_fraction=0.0,
            output_path=path,
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
        """Diffuse intensities must come back as MTZ type J, the type
        ``lunus/sf/xtraj.py`` writes, so either source reads the same way."""
        hkl = self.reflections()
        intensities = np.linspace(-0.5, 10.0, len(hkl)).astype(np.float32)
        path = tmp_path / "diffuse.mtz"

        dataset_from_intensities(
            hkl,
            intensities,
            gemmi.UnitCell(31.7, 42.3, 55.9, 90.0, 104.5, 90.0),
            gemmi.SpaceGroup("P 1 21 1"),
            output_path=path,
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
        """R-free flags are the one thing here gemmi has no equivalent for, so
        pin both branches of the switch that generates them."""
        hkl = self.reflections()
        structure_factors = np.ones(len(hkl), dtype=np.complex64)
        args = (hkl, structure_factors, gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0))

        without = tmp_path / "without.mtz"
        dataset_from_amplitudes(
            *args, gemmi.SpaceGroup("P 1"), test_fraction=0.0, output_path=without
        )
        assert not any(c.type == "I" for c in gemmi.read_mtz_file(str(without)).columns)

        with_flags = tmp_path / "with.mtz"
        dataset_from_amplitudes(
            *args, gemmi.SpaceGroup("P 1"), test_fraction=0.25, seed=7, output_path=with_flags
        )
        flags = gemmi.read_mtz_file(str(with_flags)).column_with_label("R-free-flags")
        assert flags.type == "I"
        assert set(np.unique(flags.array)) <= {0.0, 1.0}
        assert 0.0 < flags.array.mean() < 1.0

    def test_intensity_type_survives_an_unconventional_label(self, tmp_path):
        """The MTZ type is the interoperability contract, so it must stay J for
        any label. rs infers types from column names and only gives an intensity
        to names starting with "I", so a label like this would otherwise be R."""
        hkl = self.reflections(8)
        path = tmp_path / "labelled.mtz"

        dataset_from_intensities(
            hkl,
            np.ones(len(hkl), dtype=np.float32),
            gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0),
            gemmi.SpaceGroup("P 1"),
            label="DIFFUSE",
            output_path=path,
        )

        column = gemmi.read_mtz_file(str(path)).column_with_label("DIFFUSE")
        assert column.type == "J"

    def test_dataset_is_returned_without_writing(self, tmp_path):
        """Both writers are usable as builders: no output_path, no file."""
        hkl = self.reflections(8)
        cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
        space_group = gemmi.SpaceGroup("P 1")

        amplitudes = dataset_from_amplitudes(
            hkl, np.ones(len(hkl), dtype=np.complex64), cell, space_group, test_fraction=0.0
        )
        intensities = dataset_from_intensities(
            hkl, np.ones(len(hkl), dtype=np.float32), cell, space_group
        )

        assert list(tmp_path.iterdir()) == []
        assert amplitudes.index.names == ["H", "K", "L"]
        assert [str(dtype) for dtype in amplitudes.dtypes] == ["SFAmplitude", "Stddev", "Phase"]
        assert [str(dtype) for dtype in intensities.dtypes] == ["Intensity"]
