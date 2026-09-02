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

Thresholds come from measured values, recorded in each test's docstring, per
lunus's own convention. First measured 2026-08-17 on 1VME chain A at 1.8 A
(P 1 21 1, 3357 atoms, grid 96x160x160, 86499 reflections), CPU.
"""

from pathlib import Path

import numpy as np
import pytest
import torch


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Slow, but needing neither a GPU nor model weights: ~90 s of splat and FFT on
# CPU. Marked at module scope, following tests/eval/test_rscc_grid_search_script.py,
# where `slow` already covers runtime alone rather than hardware.
pytestmark = pytest.mark.slow

RESOLUTION = 1.8
SOURCE_CIF = "1vme_final.cif"


@pytest.fixture(scope="module")
def device() -> torch.device:
    """CPU: these tests check numerics, not throughput, and must run in CI."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def structure_1vme(resources_dir: Path):
    """Chain A of 1VME with hydrogens and waters stripped, as a topology + coordinates.

    Deliberately the same selection the SFcalculator reward fixtures use
    (``tests/rewards/conftest.py``), so the two engines are compared on identical
    inputs.
    """
    from sampleworks.synthetic.generate_synthetic_sf import BatchRowForMTZ
    from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations

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
    import gemmi

    meta = gemmi.read_structure(str(resources_dir / "1vme" / SOURCE_CIF))
    return meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)


def _amplitudes(atom_array, coords, cell, spacegroup, device, **kwargs):
    """Run the lunus generator's compute step and return (hkl, <F>, diffuse)."""
    from sampleworks.synthetic.generate_synthetic_sf_lunus import compute_ensemble_amplitudes

    return compute_ensemble_amplitudes(
        atom_array, coords, cell, spacegroup, RESOLUTION, device, **kwargs
    )


class TestSelfConsistency:
    """Properties that follow from the definitions, independent of any other engine."""

    def test_identical_configurations_have_zero_diffuse(self, structure_1vme, crystal_1vme, device):
        """N copies of one structure have <|F|^2> == |<F>|^2, so diffuse is zero.

        This is the sharpest check that the ensemble axis is wired correctly: if
        configurations were being summed rather than averaged, or the occupancy
        convention were doubly applied, the variance would not vanish.
        """
        atom_array, coords = structure_1vme
        cell, spacegroup = crystal_1vme
        replicated = np.repeat(coords[:1], 4, axis=0)

        _, mean_f, diffuse = _amplitudes(atom_array, replicated, cell, spacegroup, device)

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
        self, structure_1vme, crystal_1vme, device
    ):
        """<F> over N identical copies equals F of one copy.

        Guards against an ensemble weighting that scales with N.
        """
        atom_array, coords = structure_1vme
        cell, spacegroup = crystal_1vme

        _, single, _ = _amplitudes(atom_array, coords[:1], cell, spacegroup, device)
        _, replicated, _ = _amplitudes(
            atom_array, np.repeat(coords[:1], 3, axis=0), cell, spacegroup, device
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
        self, structure_1vme, crystal_1vme, device
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
        import gemmi

        atom_array, coords = structure_1vme
        cell, _ = crystal_1vme
        p1 = gemmi.SpaceGroup("P 1")
        ensemble = self._perturbed_ensemble(coords)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, p1, device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, p1, device
        )

        assert float(np.mean(diffuse)) > 0, "no diffuse signal to test invariance of"
        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\nP1 diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation < 1e-2

    def test_diffuse_is_not_invariant_under_symmetry(self, structure_1vme, crystal_1vme, device):
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
        atom_array, coords = structure_1vme
        cell, spacegroup = crystal_1vme
        assert spacegroup.hm != "P 1", "this test needs a non-trivial space group"
        ensemble = self._perturbed_ensemble(coords)

        _, _, diffuse = _amplitudes(atom_array, ensemble, cell, spacegroup, device)
        _, _, shifted = _amplitudes(
            atom_array, ensemble + np.array([1.7, -0.4, 2.3]), cell, spacegroup, device
        )

        deviation = float(np.linalg.norm(diffuse - shifted) / np.linalg.norm(diffuse))
        print(f"\n{spacegroup.hm} diffuse deviation under rigid translation: {deviation:.2e}")
        assert deviation > 0.1


class TestCrossEngineAgreement:
    """lunus versus SFcalculator on the same structure.

    Thresholds are provisional; see the module docstring.
    """

    # Measured on 1VME chain A at 1.8 A, 86499 reflections: correlation
    # 1.000000, R 0.0002, scale 1.0000 -- better than lunus's own agreement with
    # gemmi (0.999989 / 0.0077), the two engines' shared IT92 coefficients and
    # identical atom input leaving little room to disagree. Bounds are set an
    # order of magnitude looser than measured, to tolerate platform variation
    # without admitting a real regression.
    MIN_CORRELATION = 0.9999
    MAX_R_FACTOR = 0.002

    # Unlike the two above, this is NOT a measured tolerance -- it is a floor
    # that detects the engines disagreeing about which reflections are in the
    # ASU at all. The test prints the achieved coverage both ways; once those
    # are recorded from a run, tighten this to just under them.
    MIN_COVERAGE = 0.5

    @pytest.fixture(scope="class")
    @staticmethod
    def sfcalculator_amplitudes(structure_1vme, crystal_1vme, device):
        """|F| from SFcalculator on the same atoms, indexed by Miller index."""
        pytest.importorskip("SFC_Torch", reason="sfcalculator-torch not installed")
        from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi
        from SFC_Torch import SFcalculator
        from SFC_Torch.io import PDBParser

        atom_array, _ = structure_1vme
        cell, spacegroup = crystal_1vme
        gemmi_structure = atomarray_to_gemmi(atom_array, cell, spacegroup.hm)

        sfc = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=RESOLUTION,
            mode="xray",
            anomalous=False,
            set_experiment=False,
            device=device,
        )
        sfc.calc_fprotein()
        hkl = np.asarray(sfc.Hasu_array, dtype=np.int64)
        amplitude = torch.abs(sfc.Fprotein_asu).detach().cpu().numpy()
        return {tuple(h): a for h, a in zip(hkl, amplitude, strict=True)}

    def test_amplitudes_agree_with_sfcalculator(
        self, structure_1vme, crystal_1vme, device, sfcalculator_amplitudes
    ):
        """Correlation and R-factor over the reflections both engines produced.

        The two reflection sets are generated independently (gemmi's ASU here,
        SFcalculator's own there), so the comparison is over their intersection.
        A small intersection is itself a failure -- it would mean the ASU
        conventions disagree.
        """
        atom_array, coords = structure_1vme
        cell, spacegroup = crystal_1vme

        hkl, mean_f, _ = _amplitudes(atom_array, coords[:1], cell, spacegroup, device)
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
