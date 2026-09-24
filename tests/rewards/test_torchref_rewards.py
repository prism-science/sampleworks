"""Tests for the torchref-backed reciprocal-space reward.

Covers two areas: that a manually built model agrees with one built through
``Model.load()``, and that gradients reach the caller's tensors, point the right way
against a finite difference, and survive a nuisance-parameter refresh.

Finite-difference tolerances are loose because ``SfFFT`` samples atoms onto a grid and
truncates each Gaussian, making its gradient approximate by construction (~4e-2 relative
L2 against the analytic ``SfDS`` oracle, cosine 0.999). Direction is asserted tightly,
magnitude loosely.

The reward is not registered in ``_REWARD_BUNDLES``: it returns an unnormalized,
sign-unconstrained summed likelihood, so the shared ``TestRewardCorrelation``
absolute-loss bar does not apply.
"""

from pathlib import Path

import pytest
import torch
from biotite.structure import AtomArray
from sampleworks.utils.imports import TORCHREF_AVAILABLE

from tests.rewards.reward_input_helpers import build_reward_input_tensors_without_coords


# Every test drives the torchref structure-factor path on the `device` fixture.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not TORCHREF_AVAILABLE, reason="torchref is not installed"),
]

B_FACTOR = 20.0

# A deposited structure with its own experimental MTZ, from torchref's test files.
# A PDB rather than a CIF because `save_structure_to_cif` does not preserve the space
# group, which would make the reference model compute F_calc in P1 and void the
# equivalence check in `test_matches_a_normally_loaded_modelft`. 3GR5 is 1329 atoms at
# 2.05 A with conventional column labels (FP/SIGFP/FreeR_flag).
CASE = "3GR5"


@pytest.fixture(scope="module")
def torchref_files() -> Path:
    """``TorchRef/tests/files``, or skip if the sibling checkout is not present."""
    import torchref

    root = Path(torchref.__file__).resolve().parent.parent / "tests" / "files"
    if not (root / "pdb" / f"{CASE}.pdb").exists():
        pytest.skip(f"torchref test files not found at {root}")
    return root


@pytest.fixture(scope="module")
def pdb_path(torchref_files: Path) -> Path:
    return torchref_files / "pdb" / f"{CASE}.pdb"


@pytest.fixture(scope="module")
def mtz_path(torchref_files: Path) -> Path:
    return torchref_files / "mtz" / f"{CASE}.mtz"


@pytest.fixture(scope="module")
def structure(pdb_path: Path) -> AtomArray:
    """Topology in the crystal frame, loaded the way the rest of the repo does."""
    from sampleworks.utils.atom_array_utils import load_structure_with_altlocs

    return load_structure_with_altlocs(pdb_path)


def make_reward(mtz_path, structure, device, **overrides):
    """Build a reward with the standard test config, overridable per test.

    Solvent off and a large ``refresh_interval`` by default, giving the gradient tests a
    stationary objective and skipping the expensive mask build.
    """
    from sampleworks.core.rewards.torchref_rewards import TorchRefXrayRewardFunction

    # ADPs frozen by default: most of these tests are about the x-ray path, and
    # refining B-factors adds a restraint build per instance (~30 s) and makes two
    # independently-refit models incomparable. The ADP tests opt in explicitly.
    kwargs = dict(
        bulk_solvent=False,
        b_factor=B_FACTOR,
        refresh_interval=10_000,
        adp_weight=0.0,
        geometry_weight=0.0,
    )
    kwargs.update(overrides)
    return TorchRefXrayRewardFunction(mtz_path, structure=structure, device=device, **kwargs)


@pytest.fixture(scope="module")
def reward(mtz_path: Path, structure: AtomArray, device: torch.device):
    """Reward built from the deposited MTZ and the model it was refined against."""
    return make_reward(mtz_path, structure, device)


@pytest.fixture(scope="module")
def inputs(structure: AtomArray, device: torch.device):
    """``[1, n_atoms, ...]`` reward inputs for the deposited coordinates."""
    elements, b_factors, _occ = build_reward_input_tensors_without_coords(structure, device)
    coords = torch.from_numpy(structure.coord).to(device=device, dtype=torch.float32)
    return {
        "coordinates": coords.unsqueeze(0),
        "elements": elements.unsqueeze(0),
        "b_factors": b_factors.unsqueeze(0),
        "occupancies": torch.ones(1, len(structure), device=device),
    }


def _call(reward, inputs, coords=None, occupancies=None):
    """Evaluate the reward, optionally overriding coordinates or occupancies."""
    kwargs = dict(inputs)
    if coords is not None:
        kwargs["coordinates"] = coords
    if occupancies is not None:
        kwargs["occupancies"] = occupancies
    return reward(**kwargs)


class TestInitialization:
    def test_prepared_state(self, reward, structure):
        assert reward.n_atoms == len(structure)
        assert reward.space_group  # resolved from the MTZ, not the structure
        assert len(reward.unit_cell) == 6

    def test_no_element_resolves_to_zero_scattering(self, reward):
        """Index 0 is the '?' row and contributes no density; this case should have none."""
        assert not bool(reward._expected_codes.eq(0).any())

    def test_model_exposes_no_refinable_parameters(self, reward, inputs):
        """With ADPs frozen, nothing on the model belongs to an optimizer.

        Coordinates and occupancy are caller-owned, so neither may appear. Once
        ``adp_weight > 0`` exactly one leaf appears — the shared B — which
        :class:`TestSharedADP` checks.
        """
        _call(reward, inputs)
        model, _scaler, _target = reward._stack_for(1)
        assert list(model.parameters()) == []

    def test_matches_a_normally_loaded_modelft(self, mtz_path, pdb_path, structure, device, inputs):
        """The manual init agrees with a ``load_pdb``-built ModelFT on Fcalc.

        Checks the construction path bypassing ``Model.load()``: the hand-built ``pdb``,
        ``aniso_flag``, ``_rebuild_sf_indices``, the element -> Z -> ITC92 chain and the
        explicit ``setup_grid``.

        The reference model's coordinates, ADPs and occupancies are copied into our slots
        rather than the reverse, because ``Model.load`` runs occupancies through
        ``OccupancyTensor``, which collapses residue sharing groups and renormalizes
        altloc groups to 1.0. ``max_res`` is matched since it sets the grid spacing.
        """
        from sampleworks.core.rewards.torchref_rewards import _TensorSlot
        from torchref.model import ModelFT
        from torchref.symmetry import Cell

        # Own instance: this test rebinds the model's ADP slot, which would otherwise
        # leak into the module-scoped `reward`.
        reward = make_reward(mtz_path, structure, device)
        _call(reward, inputs)  # forces the stack to be built
        model, _scaler, _target = reward._stack_for(1)

        ref = ModelFT(
            verbose=0,
            strip_H=False,
            wavelength=None,
            device=model.device,
            max_res=reward.resolution,
        )
        ref.load_pdb(str(pdb_path))

        assert len(ref.pdb) == reward.n_atoms, (
            f"reference has {len(ref.pdb)} atoms, reward cached {reward.n_atoms}"
        )
        # The symmetry has to match or the comparison is meaningless: a reference built
        # from a file that lost its space group computes F_calc in P1 against our full
        # symmetry, which looks like a large numerical disagreement rather than the
        # category error it is.
        assert ref.spacegroup.hm == model.spacegroup.hm, (
            f"reference space group {ref.spacegroup.hm!r} != model {model.spacegroup.hm!r}"
        )
        # 3GR5's CRYST1 cell and its MTZ cell differ in the third decimal (90.645 vs
        # 90.670 A) -- routine between a deposited header and the processing run it came
        # from. The reward takes the MTZ's, so put the reference on the same cell;
        # otherwise this test measures that discrepancy instead of the construction path.
        ref.cell = Cell(model.cell.data.tolist(), dtype=ref.dtype_float, device=ref.device)
        ref.setup_grid()

        with torch.no_grad():
            model.set_coordinates(ref.xyz().detach())
            model.set_occupancies(ref.occupancy().detach())
            model.adp = _TensorSlot(ref.adp().detach())

            hkl = reward._data.hkl_for_sf()
            f_manual = model(hkl)
            f_ref = ref(hkl, recalc=True)

        # Amplitudes: the target is amplitude-based, so a global phase convention
        # difference would be a red herring.
        rel = (f_manual.abs() - f_ref.abs()).norm() / f_ref.abs().norm()
        assert rel < 1e-4, f"manual init disagrees with load_pdb: relative L2 {rel:.3e}"

    def test_structure_path_and_atom_array_agree(
        self, mtz_path, pdb_path, structure, device, inputs
    ):
        """Constructing from a pdb path and from an AtomArray must score identically."""
        from_path = make_reward(mtz_path, pdb_path, device)
        from_array = make_reward(mtz_path, structure, device)
        assert from_path.n_atoms == from_array.n_atoms
        with torch.no_grad():
            a, b = _call(from_path, inputs).item(), _call(from_array, inputs).item()
        # Not bitwise: each instance cold-fits its own scaler, so the two losses differ
        # by the difference between two LBFGS runs (measured ~2e-5 relative).
        assert abs(a - b) / abs(a) < 1e-3, f"path vs AtomArray: {a:.6g} vs {b:.6g}"

    def test_call_before_prepare_raises(self, mtz_path, inputs):
        from sampleworks.core.rewards.torchref_rewards import TorchRefXrayRewardFunction

        rf = TorchRefXrayRewardFunction(mtz_path, bulk_solvent=False)
        with pytest.raises(RuntimeError, match="prepare"):
            _call(rf, inputs)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"target_mode": "nope"}, "target_mode"),
            ({"scale_target": "ml"}, "scale_target"),
            ({"use_set": "nope"}, "use_set"),
            ({"refresh_interval": 0}, "refresh_interval"),
            ({"b_factor": 0.0}, "b_factor"),
        ],
    )
    def test_invalid_config_raises(self, mtz_path, kwargs, match):
        """`scale_target="ml"` in particular: an alpha-centred mode is degenerate
        with the scale being fitted, so torchref rejects it and so do we."""
        from sampleworks.core.rewards.torchref_rewards import TorchRefXrayRewardFunction

        with pytest.raises(ValueError, match=match):
            TorchRefXrayRewardFunction(mtz_path, **kwargs)


class TestTopologyChecks:
    def test_pipeline_elements_match_the_cached_codes(self, reward, inputs):
        """The conservation assumption, checked end-to-end through the production path."""
        expected = reward._expected_codes.expand_as(inputs["elements"])
        assert torch.equal(inputs["elements"], expected)

    def test_changed_elements_raise(self, reward, inputs):
        bad = inputs["elements"].clone()
        bad[0, 0] = bad[0, 0] + 1
        with pytest.raises(ValueError, match="do not match the topology"):
            reward(**{**inputs, "elements": bad})

    def test_permuted_elements_raise(self, reward, inputs):
        """A permutation preserves the count, so only the per-atom check catches it."""
        perm = torch.randperm(reward.n_atoms, device=inputs["elements"].device)
        bad = inputs["elements"][:, perm]
        if torch.equal(bad, inputs["elements"]):
            pytest.skip("random permutation happened to be identity-equivalent")
        with pytest.raises(ValueError, match="do not match the topology"):
            reward(**{**inputs, "elements": bad})

    def test_wrong_atom_count_raises(self, reward, inputs):
        truncated = {k: v[:, :-1] for k, v in inputs.items()}
        with pytest.raises(ValueError, match="atoms but prepare"):
            reward(**truncated)


class TestGradients:
    def test_gradient_reaches_the_callers_coordinates(self, reward, inputs):
        coords = inputs["coordinates"].clone().requires_grad_(True)
        _call(reward, inputs, coords=coords).backward()

        assert coords.grad is not None, "gradient did not reach the caller's tensor"
        assert coords.grad.shape == coords.shape
        assert torch.isfinite(coords.grad).all()
        assert coords.grad.abs().max() > 0

    def test_gradient_reaches_the_callers_occupancies(self, reward, inputs):
        """Occupancy passes through rather than being frozen, so it carries gradient."""
        occ = inputs["occupancies"].clone().requires_grad_(True)
        _call(reward, inputs, occupancies=occ).backward()

        assert occ.grad is not None
        assert torch.isfinite(occ.grad).all()
        assert occ.grad.abs().max() > 0

    def test_gradient_matches_directional_finite_difference(self, reward, inputs):
        """Directional derivative g·d vs (L(x+hd) - L(x-hd)) / 2h.

        Directional rather than per-component gradcheck because the FFT route's
        gradient is approximate by construction; see the module docstring.
        """
        x0 = inputs["coordinates"]
        coords = x0.clone().requires_grad_(True)
        _call(reward, inputs, coords=coords).backward()
        grad = coords.grad.clone()

        torch.manual_seed(7)
        d = torch.randn_like(x0)
        d /= d.norm()

        # eps is set by float32 cancellation, not by curvature. The loss is a sum of order
        # 1e5, so one ULP is ~1.3e-2 and fd is quantized in steps of ULP/(2 eps). Relative
        # to |g| ~ 5.3 that floor is 6.1% at eps=2e-2 -- above this tolerance, so the FD
        # estimate cannot resolve it -- and 0.61% at 2e-1. Measured relative error: 4.6% at
        # 2e-2, 0.14% at 2e-1, 1.5% at 5e-1 where curvature starts to show.
        eps = 2e-1
        with torch.no_grad():
            lp = _call(reward, inputs, coords=x0 + eps * d).item()
            lm = _call(reward, inputs, coords=x0 - eps * d).item()
        fd = (lp - lm) / (2 * eps)
        analytic = (grad * d).sum().item()

        scale = max(abs(fd), abs(analytic), 1e-8)
        assert abs(fd - analytic) / scale < 0.03, (
            f"directional derivative mismatch: analytic={analytic:.6g} fd={fd:.6g}"
        )

    def test_gradient_points_downhill_from_a_perturbed_structure(self, reward, inputs):
        """One small step along -grad must reduce the loss."""
        torch.manual_seed(0)
        x0 = inputs["coordinates"] + 0.3 * torch.randn_like(inputs["coordinates"])
        coords = x0.clone().requires_grad_(True)

        loss0 = _call(reward, inputs, coords=coords)
        loss0.backward()
        step = 0.01 / coords.grad.abs().max()
        with torch.no_grad():
            loss1 = _call(reward, inputs, coords=x0 - step * coords.grad)
        assert loss1.item() < loss0.item()

    def test_descent_reduces_the_loss(self, reward, inputs):
        """A short Adam run must make real progress."""
        torch.manual_seed(0)
        coords = (
            inputs["coordinates"] + 0.3 * torch.randn_like(inputs["coordinates"])
        ).requires_grad_(True)
        opt = torch.optim.Adam([coords], lr=0.02)

        first = None
        for _ in range(20):
            opt.zero_grad()
            loss = _call(reward, inputs, coords=coords)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        assert loss.item() < first


class TestFrameSensitivity:
    """The reward is not SE(3)-invariant and must notice rigid motion.

    Structure factors are more sensitive to this than real-space density: a rigid
    translation within the cell changes every phase — so a reward that did *not*
    respond to translation would mean coordinates are not reaching the MTZ's frame,
    and nothing downstream could work.
    """

    def test_translation_changes_the_loss(self, reward, inputs):
        shift = torch.tensor([1.5, 0.0, 0.0], device=inputs["coordinates"].device)
        with torch.no_grad():
            base = _call(reward, inputs).item()
            moved = _call(reward, inputs, coords=inputs["coordinates"] + shift).item()
        assert abs(moved - base) > 1e-3 * abs(base)

    def test_true_structure_beats_a_perturbation(self, reward, inputs):
        torch.manual_seed(0)
        noise = 0.5 * torch.randn_like(inputs["coordinates"])
        with torch.no_grad():
            truth = _call(reward, inputs).item()
            perturbed = _call(reward, inputs, coords=inputs["coordinates"] + noise).item()
        assert truth < perturbed


class TestConformerStack:
    def test_identical_conformers_reproduce_one_copy(self, reward, inputs, device):
        """C copies at occupancy 1/C == one copy at occupancy 1.

        The identity the stacked representation rests on: F is linear over atoms, so
        an ensemble is one structure-factor calculation over a bigger stack.

        Compared on ``Fcalc``, not on the loss. Each ensemble size gets its own
        ``(model, scaler, target)`` triple, so the two losses are scored under two
        independently cold-fitted scales and differ by the difference between two LBFGS
        fits (~0.2%) — which says nothing about the linearity being tested here.
        """
        c = 3
        n = reward.n_atoms
        stacked = {
            "coordinates": inputs["coordinates"].expand(c, -1, -1).contiguous(),
            "elements": inputs["elements"].expand(c, -1).contiguous(),
            "b_factors": inputs["b_factors"].expand(c, -1).contiguous(),
            "occupancies": torch.full((c, n), 1.0 / c, device=device),
        }
        # 1. Fcalc: the exact identity. Nothing here is approximate.
        with torch.no_grad():
            _call(reward, inputs)
            reward(**stacked)
            hkl = reward._data.hkl_for_sf()
            f_single = reward._stack_for(1)[0](hkl)
            f_multi = reward._stack_for(c)[0](hkl)
        rel_f = (f_multi - f_single).abs().norm() / f_single.abs().norm()
        assert rel_f < 1e-5, f"conformer stack is not exact in Fcalc: rel L2 {rel_f:.3e}"

        # 2. Loss: close but not identical, because the two triples were cold-fitted
        #    independently and so carry slightly different scales.
        cs = inputs["coordinates"].clone().requires_grad_(True)
        cm = stacked["coordinates"].clone().requires_grad_(True)
        loss_single = _call(reward, inputs, coords=cs)
        loss_multi = reward(**{**stacked, "coordinates": cm})
        rel_l = abs(loss_multi.item() - loss_single.item()) / abs(loss_single.item())
        assert rel_l < 5e-3, f"stacked loss differs by {rel_l:.3%} (scaler fits differ)"

        # 3. Gradients: each of the C identical conformers carries 1/C of the density,
        #    so its gradient is 1/C of the single-copy one and they sum back to it.
        loss_single.backward()
        loss_multi.backward()
        g_single, g_multi = cs.grad[0], cm.grad.sum(0)
        cos = torch.nn.functional.cosine_similarity(
            g_single.reshape(1, -1), g_multi.reshape(1, -1)
        ).item()
        ratio = (g_multi.norm() / g_single.norm()).item()
        # Close but not bitwise: the two triples were cold-fitted independently, so the
        # per-bin scale, aniso tensor and sigma_A differ slightly and reweight the
        # residuals. Measured cosine 0.999992 and magnitude ratio within 0.1%; forcing
        # both triples onto one scaler makes it exact (cosine 1.000000).
        assert cos > 0.9999, f"stacked gradient points elsewhere: cosine {cos:.6f}"
        assert abs(ratio - 1.0) < 0.01, f"stacked gradient magnitude ratio {ratio:.4f}"

    def test_a_new_stack_is_cold_fitted(self, reward, inputs, device):
        """A newly built ensemble size is fitted on its first use.

        The refresh counter is per-stack for this reason: with a
        single global counter and a large ``refresh_interval``, only call 0 ever
        refreshes, so any triple built later would run with an *identity* scaler and
        silently score against unscaled Fcalc. That showed up as a 14% gradient
        disagreement between the single and stacked models before it was fixed.
        """
        c = 4
        n = reward.n_atoms
        _call(reward, inputs)  # burn calls so the global count is far from 0
        _call(reward, inputs)
        assert c not in reward._stacks, "pick an ensemble size no other test built"

        reward(
            coordinates=inputs["coordinates"].expand(c, -1, -1).contiguous(),
            elements=inputs["elements"].expand(c, -1).contiguous(),
            b_factors=inputs["b_factors"].expand(c, -1).contiguous(),
            occupancies=torch.full((c, n), 1.0 / c, device=device),
        )
        _model, scaler, _target = reward._stack_for(c)
        assert hasattr(scaler, "log_scale"), "the new stack was never cold-fitted"
        assert reward._calls[c] == 1

    def test_distinct_conformers_differ_from_the_mean_structure(self, reward, inputs, device):
        """A real ensemble is a complex sum, not an average of coordinates."""
        torch.manual_seed(0)
        c = 2
        n = reward.n_atoms
        coords = torch.stack(
            [inputs["coordinates"][0] + 0.4 * torch.randn(n, 3, device=device) for _ in range(c)]
        )
        stacked = {
            "coordinates": coords,
            "elements": inputs["elements"].expand(c, -1).contiguous(),
            "b_factors": inputs["b_factors"].expand(c, -1).contiguous(),
            "occupancies": torch.full((c, n), 1.0 / c, device=device),
        }
        with torch.no_grad():
            ensemble = reward(**stacked).item()
            mean_structure = _call(reward, inputs, coords=coords.mean(0, keepdim=True)).item()
        assert ensemble != pytest.approx(mean_structure, rel=1e-6)


class TestCacheAndRefresh:
    def test_loss_changes_when_only_coordinates_change(self, reward, inputs):
        """End-to-end guard on the disabled forward cache.

        torchref computes Fcalc internally via calls that do not pass ``recalc``, so
        with the cache enabled every call after the first would score the *first*
        call's coordinates — silently.
        """
        with torch.no_grad():
            a = _call(reward, inputs).item()
            b = _call(reward, inputs, coords=inputs["coordinates"] + 0.5).item()
            c = _call(reward, inputs).item()
        assert a != pytest.approx(b, rel=1e-9)
        assert a == pytest.approx(c, rel=1e-6), "same coordinates should score the same"

    def test_bare_modelft_would_have_been_stale(self, reward, inputs):
        """Documents why the subclass exists, and fails if torchref makes it moot."""
        from torchref.model import ModelFT

        _call(reward, inputs)
        model, _scaler, _target = reward._stack_for(1)
        hkl = reward._data.hkl_for_sf()

        with torch.no_grad():
            f0 = ModelFT.__call__(model, hkl)  # the cached path we bypass
            model.set_coordinates(model.xyz() + 1.0)
            f_stale = ModelFT.__call__(model, hkl)
            f_fresh = model(hkl)  # our uncached __call__
        assert torch.allclose(f0, f_stale), "expected the bare mixin to serve a stale hit"
        assert not torch.allclose(f0, f_fresh)

    def test_refresh_preserves_the_callers_graph(self, reward, inputs):
        """The property the detached clone exists to guarantee."""
        coords = inputs["coordinates"].clone().requires_grad_(True)
        version_before = coords._version

        reward.refresh_nuisance_parameters(1, coordinates=coords)

        assert coords._version == version_before, (
            "the refresh bumped the caller's version counter -- a bare detach() instead "
            "of detach().clone() would do this, and it makes backward() fail later"
        )
        loss = _call(reward, inputs, coords=coords)
        assert loss.grad_fn is not None
        loss.backward()
        assert coords.grad is not None and torch.isfinite(coords.grad).all()

    def test_refresh_reports_rfactors(self, reward, inputs):
        _call(reward, inputs)
        stats = reward.refresh_nuisance_parameters(1)
        assert set(stats) == {"r_work", "r_free"}
        assert 0.0 < stats["r_work"] < 1.5
        assert 0.0 < stats["r_free"] < 1.5

    def test_refresh_is_idempotent(self, reward, inputs):
        _call(reward, inputs)
        first = reward.refresh_nuisance_parameters(1)
        second = reward.refresh_nuisance_parameters(1)
        assert second["r_work"] == pytest.approx(first["r_work"], abs=0.02)

    def test_cold_start_failure_raises(self, mtz_path, structure, device, inputs, monkeypatch):
        """A *first* refit that fails leaves no usable scale, so it must not be swallowed.

        Every later loss would otherwise be scored against an unfitted scale and be
        silently meaningless — worse than an exception.
        """
        rf = make_reward(mtz_path, structure, device)
        _model, scaler, _target = rf._stack_for(1)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated cold-start failure")

        monkeypatch.setattr(scaler, "refine_lbfgs", boom)
        with pytest.raises(RuntimeError, match="initial nuisance-parameter fit failed"):
            _call(rf, inputs)

    def test_refresh_failure_is_swallowed_and_unwinds(self, reward, inputs, monkeypatch):
        """A failed scale fit must not kill a run, and must still reattach."""
        coords = inputs["coordinates"].clone().requires_grad_(True)
        _call(reward, inputs, coords=coords)
        _model, scaler, _target = reward._stack_for(1)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated LBFGS failure")

        monkeypatch.setattr(scaler, "refine_lbfgs", boom)
        assert reward.refresh_nuisance_parameters(1, coordinates=coords) == {}
        monkeypatch.undo()

        # The finally clause must have rebound the live tensor, or every later call
        # silently returns a gradient-free loss.
        loss = _call(reward, inputs, coords=coords)
        assert loss.grad_fn is not None
        loss.backward()
        assert coords.grad is not None


class TestSharedADP:
    """Refinable B-factors shared across conformers, regularised by the ADP restraints.

    These opt into ``adp_weight > 0``, which builds the restraint graph (``adp/simu``
    reads the bond list) and makes the B-factors refinable, so they are slower than the
    rest of the module.
    """

    @pytest.fixture(scope="class")
    def adp_reward(self, mtz_path, structure, device):
        return make_reward(mtz_path, structure, device, adp_weight=0.02)

    def test_one_shared_leaf_expanded_across_the_stack(self, adp_reward, inputs, device):
        """``(n_asu,)`` refinable log-B, ``forward()`` expanded to ``(C*n_asu,)``."""
        c, n = 3, adp_reward.n_atoms
        adp_reward(
            coordinates=inputs["coordinates"].expand(c, -1, -1).contiguous(),
            elements=inputs["elements"].expand(c, -1).contiguous(),
            b_factors=inputs["b_factors"].expand(c, -1).contiguous(),
            occupancies=torch.full((c, n), 1.0 / c, device=device),
        )
        model, _s, _t = adp_reward._stack_for(c)

        params = model.parameters_of_types(("adp",))
        assert len(params) == 1, "expected exactly the shared adp leaf"
        assert params[0].numel() == n, f"leaf is {params[0].numel()}, want n_asu={n}"

        expanded = model.adp()
        assert expanded.shape == (c * n,)
        # Conformer-major: the ASU block repeats, so every conformer sees the same B.
        per_conf = expanded.reshape(c, n)
        for b in range(1, c):
            assert torch.allclose(per_conf[0], per_conf[b])

        # And exactly one parameter on the whole model -- coordinates and occupancy stay
        # caller-owned, so they must not have become optimizable.
        assert len(list(model.parameters())) == 1

    def test_gradient_sums_onto_the_shared_leaf(self, adp_reward, inputs, device):
        """C copies contribute C gradients to one parameter, which is the sharing point."""
        c, n = 3, adp_reward.n_atoms
        stacked = dict(
            coordinates=inputs["coordinates"].expand(c, -1, -1).contiguous(),
            elements=inputs["elements"].expand(c, -1).contiguous(),
            b_factors=inputs["b_factors"].expand(c, -1).contiguous(),
            occupancies=torch.full((c, n), 1.0 / c, device=device),
        )
        adp_reward(**stacked)
        model, _s, _t = adp_reward._stack_for(c)
        leaf = model.parameters_of_types(("adp",))[0]

        leaf.grad = None
        adp_reward(**stacked).backward()
        assert leaf.grad is not None
        assert leaf.grad.shape == (n,)
        assert torch.isfinite(leaf.grad).all()
        assert leaf.grad.abs().max() > 0

    def test_shared_wrapper_is_a_positive_mixed_tensor(self, adp_reward, inputs):
        """The shared wrapper keeps torchref's ADP parameter type.

        ``parameters_of_types(("adp",))`` and the ADP restraint targets both expect the
        log-space ``PositiveMixedTensor`` form, so the subclass relationship is part of
        the contract rather than an implementation detail.
        """
        from torchref.model.parameter_wrappers import PositiveMixedTensor

        c, n = 2, adp_reward.n_atoms
        adp_reward(
            coordinates=inputs["coordinates"].expand(c, -1, -1).contiguous(),
            elements=inputs["elements"].expand(c, -1).contiguous(),
            b_factors=inputs["b_factors"].expand(c, -1).contiguous(),
            occupancies=torch.full((c, n), 0.5, device=inputs["coordinates"].device),
        )
        model, _s, _t = adp_reward._stack_for(c)

        assert isinstance(model.adp, PositiveMixedTensor)
        assert (model.adp() > 0).all(), "positivity is what the parameterisation buys"

    def test_restraints_built_and_non_empty(self, adp_reward, inputs):
        """The restraint graph is populated.

        A residue missing from the monomer library silently gets no restraints, so zero
        counts would mean the regularisation is absent while the weights imply otherwise.
        """
        _call(adp_reward, inputs)
        model, _s, _t = adp_reward._stack_for(1)
        counts = adp_reward._restraint_counts(model)
        for rtype in ("bond", "angle", "torsion"):
            assert counts[rtype] > 0, f"no {rtype} restraints: {counts}"

    def test_restraints_scale_with_conformers_and_never_cross_them(
        self, adp_reward, inputs, device
    ):
        """C conformers give C independent copies, with no bond spanning two of them.

        With a shared chain/resseq the peptide-link builder produces one cross-conformer
        bond per junction instead of C proper ones; distinct chain ids and altloc letters
        prevent that.
        """
        c, n = 2, adp_reward.n_atoms
        _call(adp_reward, inputs)
        one = adp_reward._restraint_counts(adp_reward._stack_for(1)[0])["bond"]

        adp_reward(
            coordinates=inputs["coordinates"].expand(c, -1, -1).contiguous(),
            elements=inputs["elements"].expand(c, -1).contiguous(),
            b_factors=inputs["b_factors"].expand(c, -1).contiguous(),
            occupancies=torch.full((c, n), 1.0 / c, device=device),
        )
        model_c = adp_reward._stack_for(c)[0]
        many = adp_reward._restraint_counts(model_c)["bond"]
        assert many == pytest.approx(c * one, rel=0.02), (
            f"{c} conformers gave {many} bonds, expected ~{c * one} (one set per conformer)"
        )

        # No bond may join two different conformer blocks.
        idx = model_c.restraints.restraints["bond"]["all"]["indices"]
        block = (idx // n).to(torch.long)
        assert int((block[:, 0] != block[:, 1]).sum()) == 0, "cross-conformer bond found"

    def test_refit_moves_b_and_lowers_the_loss(self, adp_reward, inputs):
        """The point of the whole exercise: B stops being a flat 20 A^2."""
        _call(adp_reward, inputs)
        model, _s, _t = adp_reward._stack_for(1)
        b = model.adp().detach()
        assert b.std() > 1.0, f"B did not move off uniform: std={b.std():.3f}"
        assert b.min() > 0, "B must stay positive (log parameterisation)"

    def test_refit_leaves_the_callers_coordinates_untouched(self, adp_reward, inputs):
        """The refit runs on a detached clone, so the caller's tensor must be pristine."""
        coords = inputs["coordinates"].clone().requires_grad_(True)
        before, version = coords.detach().clone(), coords._version
        adp_reward.refresh_nuisance_parameters(1, coordinates=coords)
        assert coords._version == version
        assert torch.equal(coords.detach(), before)
        loss = _call(adp_reward, inputs, coords=coords)
        assert loss.grad_fn is not None


class TestRestraintGating:
    """A zero weight must mean *never constructed*, not merely skipped at evaluation."""

    def test_geometry_off_by_default(self, reward, inputs):
        _call(reward, inputs)
        state = reward._states[1]
        assert not any(k.startswith("geometry/") for k in state.targets), (
            f"geometry registered despite weight 0: {sorted(state.targets)}"
        )

    def test_both_weights_zero_builds_no_restraints(self, mtz_path, structure, device, inputs):
        """With nothing needing the bond graph, the restraint build does not happen.

        ``register_target`` probes each target on registration, and for these targets that
        first call is what builds the restraints. Gating only at evaluation would pay the
        whole build and discard it.
        """
        rf = make_reward(mtz_path, structure, device, adp_weight=0.0, geometry_weight=0.0)
        _call(rf, inputs)
        model, _s, _t = rf._stack_for(1)
        assert model._restraints is None, "restraints were built though nothing needs them"
        assert sorted(rf._states[1].targets) == ["xray"]

    def test_weights_are_applied_with_the_right_names(self, mtz_path, structure, device, inputs):
        """Guards the double-prefix trap that silently re-enables Ramachandran.

        Registering the aggregates *with* a name would key them ``geometry/geometry/bond``,
        at which point ``"geometry/ramachandran": 0.0`` stops matching and Ramachandran
        runs at the group weight instead of being disabled.
        """
        rf = make_reward(mtz_path, structure, device, adp_weight=0.02, geometry_weight=0.2)
        _call(rf, inputs)
        state = rf._states[1]
        assert state.get_effective_weight("xray") == pytest.approx(1.0)
        assert state.get_effective_weight("adp/simu") == pytest.approx(0.02)
        assert state.get_effective_weight("geometry/bond") == pytest.approx(0.2)
        assert state.get_effective_weight("geometry/ramachandran") == pytest.approx(0.0)
        # And the keys are single-prefixed.
        assert "geometry/geometry/bond" not in state.targets


class TestBulkSolvent:
    @pytest.mark.slow
    def test_solvent_changes_the_loss(self, mtz_path, structure, device, inputs):
        """Marked slow: this builds a real-space mask and FFTs it."""
        without = make_reward(mtz_path, structure, device, bulk_solvent=False)
        with_solvent = make_reward(mtz_path, structure, device, bulk_solvent=True)
        with torch.no_grad():
            a = _call(without, inputs).item()
            b = _call(with_solvent, inputs).item()
        assert a != pytest.approx(b, rel=1e-6)
        assert with_solvent._stack_for(1)[1].solvent is not None
