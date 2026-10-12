"""Tests specific to the combined Bragg + diffuse reward.

The generic reward tests (output shape, determinism, recovery and gradient flow for
a single conformer, batch handling) live in ``test_reward_function_contract.py``,
where this reward is registered as the ``diffuse_bragg`` bundle. That bundle scores
the Bragg term alone, because the contract tests use one conformer and the diffuse
term, ``<|F|²> − |<F>|²`` over the ensemble, is identically zero for one conformer.

This file tests what needs a real ensemble, using the two-configuration fixtures
in ``conftest.py`` (#495 tracks moving these checks into the contract): the loss
is ~0 for the true ensemble, rises as the ensemble is displaced, and puts a
gradient on every configuration, each at every mixture of the two terms; the
diffuse gradient points downhill; and a sign-flipped diffuse pattern is not
mistaken for a match. It also covers the options this reward alone has: the
convex weight, separate Bragg and diffuse reflection lists, R-free handling and
bulk solvent.

The targets are written by the lunus generator in-process rather than read from a
checked-in file, so the fixtures cannot drift from the generator that made them.
"""

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs
import torch
from sampleworks.core.rewards.protocol import RewardInputs

from tests.rewards.conftest import DiffuseBraggTargets


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Deliberately unmarked: these need neither a GPU nor model weights, and the module
# runs in well under a minute on CPU. Marking it would hide the only coverage of the
# ensemble terms from `pixi run -e <env> tests`.


def prepared_reward(targets: DiffuseBraggTargets, atom_array, weight, **kwargs):
    """A reward on both ensemble targets at ``bragg_weight=weight``, prepared on ``atom_array``.

    Prepared the way the sampling loop does it, through single-conformer
    ``RewardInputs``. Keyword arguments go to ``DiffuseBraggRewardFunction``.
    """
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

    reward = DiffuseBraggRewardFunction(
        bragg_target=targets.bragg, diffuse_target=targets.diffuse, bragg_weight=weight, **kwargs
    )
    reward.prepare(RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu")
    return reward


def score(reward, atom_array, coords, requires_grad=False):
    """Evaluate the reward on ``(n_configs, n_atoms, 3)`` coordinates as the loop would.

    Occupancies are divided by the configuration count because the reward undoes
    the ``1/N`` convention ``RewardInputs`` imposes. Returns the loss and the
    coordinate tensor, which carries ``.grad`` after ``backward()``.
    """
    n = coords.shape[0]
    x = torch.as_tensor(coords, dtype=torch.float32)
    if requires_grad:
        x = x.detach().clone().requires_grad_(True)
    occ = torch.as_tensor(np.tile(np.asarray(atom_array.occupancy, dtype=np.float32) / n, (n, 1)))
    dummy = torch.zeros_like(occ)
    return reward(x, dummy.long(), dummy, occ), x


# ---------------------------------------------------------------------------
# Ensemble terms, at each mixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_recovers_the_ensemble_it_was_generated_from(
    diffuse_bragg_ensemble, diffuse_bragg_rewards, weight
):
    """~0 loss when the model is the two-configuration truth, at every mixture.

    Measured on 1vme (2 configurations, 86499 reflections): 2e-14 pure Bragg,
    6e-10 pure diffuse. Bragg reaches float32 epsilon because that is what the
    MTZ stores; diffuse is looser because the anisotropic component is what
    survives subtracting a large radial background, which amplifies the relative
    error of the intensities it came from.
    """
    atom_array, coords = diffuse_bragg_ensemble
    loss, _ = score(diffuse_bragg_rewards[weight], atom_array, coords)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_loss_increases_with_ensemble_displacement(
    diffuse_bragg_ensemble, diffuse_bragg_rewards, weight
):
    """Monotonic in how far the ensemble has been moved from the truth.

    The displacement is a fixed random direction scaled up, so the sequence
    probes one path away from the minimum rather than several unrelated points.
    A term dominated by its own numerical noise would fail here while still
    passing the recovery test above.
    """
    atom_array, coords = diffuse_bragg_ensemble
    direction = np.random.default_rng(0).normal(size=coords.shape)
    losses = [
        score(diffuse_bragg_rewards[weight], atom_array, coords + amplitude * direction)[0].item()
        for amplitude in (0.0, 0.02, 0.05, 0.1, 0.2)
    ]

    assert all(np.isfinite(losses))
    assert losses == sorted(losses), f"loss is not monotonic in displacement: {losses}"
    assert losses[-1] > 1e3 * max(losses[0], 1e-12), (
        f"a 0.2 A displacement barely moved the loss: {losses}"
    )


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_every_configuration_receives_a_gradient(
    diffuse_bragg_ensemble, diffuse_bragg_rewards, weight
):
    """backward() puts finite, non-zero gradients on every configuration.

    Checked per configuration because an ensemble axis that collapsed -- all
    members receiving the same gradient, or only the first receiving any --
    would still produce a plausible scalar loss.
    """
    atom_array, coords = diffuse_bragg_ensemble
    displaced = coords + 0.1 * np.random.default_rng(1).normal(size=coords.shape)
    loss, x = score(diffuse_bragg_rewards[weight], atom_array, displaced, requires_grad=True)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for i in range(coords.shape[0]):
        assert x.grad[i].abs().max() > 0, f"configuration {i} received no gradient"


def test_diffuse_gradient_points_downhill(diffuse_bragg_ensemble, diffuse_bragg_rewards):
    """A small step along -grad of the pure-diffuse loss lowers it.

    The contract checks the sign for Bragg at one conformer; the diffuse term's
    gradient only exists for an ensemble, and a wrong sign there would drive
    guidance away from the data while every magnitude check passed.
    """
    atom_array, coords = diffuse_bragg_ensemble
    reward = diffuse_bragg_rewards[0.0]
    displaced = coords + 0.1 * np.random.default_rng(2).normal(size=coords.shape)
    loss, x = score(reward, atom_array, displaced, requires_grad=True)
    loss.backward()

    step = 1e-3 / x.grad.abs().max().item()
    lowered, _ = score(reward, atom_array, displaced - step * x.grad.detach().numpy())

    assert lowered.item() < loss.item(), (
        f"stepping along -grad raised the loss: {loss.item():.6e} -> {lowered.item():.6e}"
    )


def test_anticorrelated_diffuse_is_not_a_match(
    diffuse_bragg_ensemble, diffuse_bragg_targets, tmp_path
):
    """The fitted scale is floored at zero, so a sign-flipped pattern scores ~1, not ~0.

    The anisotropic diffuse component is signed on both sides. An unconstrained
    least-squares scale would go negative for an anticorrelated calc, flip it onto
    the target and report a perfect fit.
    """
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

    negated = rs.read_mtz(str(diffuse_bragg_targets.diffuse))
    negated["ID"] = -negated["ID"]
    path = tmp_path / "diffuse_negated.mtz"
    negated.write_mtz(str(path))

    atom_array, coords = diffuse_bragg_ensemble
    reward = DiffuseBraggRewardFunction(diffuse_target=path, bragg_weight=0.0)
    reward.prepare(RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu")
    loss, _ = score(reward, atom_array, coords)

    assert loss.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# The convex weight
# ---------------------------------------------------------------------------


class TestWeightHandling:
    """The convex weight: what it requires, what it rejects, how a callable is read."""

    def test_nonzero_weight_requires_its_target(self, diffuse_bragg_targets):
        """A mixture needs both targets; the error names the missing one."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        with pytest.raises(ValueError, match="bragg_target is required"):
            DiffuseBraggRewardFunction(
                diffuse_target=diffuse_bragg_targets.diffuse, bragg_weight=0.5
            )
        with pytest.raises(ValueError, match="diffuse_target is required"):
            DiffuseBraggRewardFunction(bragg_target=diffuse_bragg_targets.bragg, bragg_weight=0.5)

    def test_numpy_scalar_weight_is_a_constant(self, diffuse_bragg_ensemble, diffuse_bragg_targets):
        """A np.float32 weight is stored as a float, not mistaken for a callable."""
        atom_array, coords = diffuse_bragg_ensemble
        reward = prepared_reward(diffuse_bragg_targets, atom_array, np.float32(0.5))

        assert reward.bragg_weight == 0.5
        loss, _ = score(reward, atom_array, coords)
        assert torch.isfinite(loss)

    def test_weight_outside_the_unit_interval_is_rejected(self, diffuse_bragg_targets):
        """The mixture is convex, so a weight outside [0, 1] is refused at construction."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        with pytest.raises(ValueError, match="convex"):
            DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg,
                diffuse_target=diffuse_bragg_targets.diffuse,
                bragg_weight=1.5,
            )

    def test_callable_weight_without_a_time_is_a_clear_error(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets
    ):
        """RewardFunctionProtocol carries no t, so nothing sets current_time yet."""
        atom_array, coords = diffuse_bragg_ensemble
        reward = prepared_reward(diffuse_bragg_targets, atom_array, lambda t: t)

        with pytest.raises(RuntimeError, match="current_time"):
            score(reward, atom_array, coords)

    def test_callable_weight_is_evaluated_at_current_time(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, diffuse_bragg_rewards
    ):
        """A step-function weight reproduces the pure-Bragg and pure-diffuse rewards exactly."""
        atom_array, coords = diffuse_bragg_ensemble
        reward = prepared_reward(
            diffuse_bragg_targets, atom_array, lambda t: 1.0 if t > 0.5 else 0.0
        )
        displaced = coords + 0.05 * np.random.default_rng(3).normal(size=coords.shape)

        reward.current_time = 0.9
        at_high_t, _ = score(reward, atom_array, displaced)
        reward.current_time = 0.1
        at_low_t, _ = score(reward, atom_array, displaced)

        bragg_only, _ = score(diffuse_bragg_rewards[1.0], atom_array, displaced)
        diffuse_only, _ = score(diffuse_bragg_rewards[0.0], atom_array, displaced)
        assert at_high_t.item() == pytest.approx(bragg_only.item(), rel=1e-5)
        assert at_low_t.item() == pytest.approx(diffuse_only.item(), rel=1e-5)
        assert bragg_only.item() != pytest.approx(diffuse_only.item())


def test_prepare_without_an_atom_array_is_a_clear_error(
    diffuse_bragg_ensemble, diffuse_bragg_targets
):
    """The topology cannot be recovered from the tensors alone."""
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

    atom_array, _ = diffuse_bragg_ensemble
    inputs = RewardInputs.from_atom_array(atom_array, ensemble_size=1)
    inputs.atom_array = None
    reward = DiffuseBraggRewardFunction(
        bragg_target=diffuse_bragg_targets.bragg,
        diffuse_target=diffuse_bragg_targets.diffuse,
        bragg_weight=0.5,
    )

    with pytest.raises(ValueError, match="atom_array"):
        reward.prepare(inputs)


# ---------------------------------------------------------------------------
# R-free handling
# ---------------------------------------------------------------------------


class TestFreeReflections:
    """exclude_free_reflections drops the test set from the Bragg term."""

    def test_test_set_is_dropped_from_the_bragg_list(self, diffuse_bragg_targets):
        """The Bragg list shrinks by the flagged count; the diffuse list is untouched."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        targets = diffuse_bragg_targets
        both = {"bragg_target": targets.bragg, "diffuse_target": targets.diffuse}
        with_all = DiffuseBraggRewardFunction(bragg_weight=0.5, **both)
        working = DiffuseBraggRewardFunction(
            bragg_weight=0.5, exclude_free_reflections=True, **both
        )

        assert len(with_all._bragg_hkl) == targets.n_reflections
        assert len(working._bragg_hkl) == targets.n_reflections - targets.n_free
        assert len(working._diffuse_hkl) == targets.n_reflections

    def test_wrong_convention_warns(self, diffuse_bragg_targets, caplog):
        """test_set_value=0 on a Phenix-convention file keeps only the 10% test set."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        reward = DiffuseBraggRewardFunction(
            bragg_target=diffuse_bragg_targets.bragg,
            bragg_weight=1.0,
            exclude_free_reflections=True,
            test_set_value=0,
        )

        assert len(reward._bragg_hkl) == diffuse_bragg_targets.n_free
        assert "CCP4-convention" in caplog.text

    def test_reflection_with_a_missing_flag_stays_in_the_working_set(
        self, diffuse_bragg_targets, tmp_path
    ):
        """A null flag puts a reflection in neither set; it is kept in the working set."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        ds = rs.read_mtz(str(diffuse_bragg_targets.bragg))
        first_free = int(np.flatnonzero(ds["R-free-flags"].to_numpy() == 1)[0])
        ds.iloc[first_free, ds.columns.get_loc("R-free-flags")] = pd.NA
        path = tmp_path / "bragg_null_flag.mtz"
        ds.write_mtz(str(path))

        reward = DiffuseBraggRewardFunction(
            bragg_target=path, bragg_weight=1.0, exclude_free_reflections=True
        )

        n = diffuse_bragg_targets.n_reflections - diffuse_bragg_targets.n_free + 1
        assert len(reward._bragg_hkl) == n

    def test_missing_flag_column_is_a_clear_error(self, diffuse_bragg_targets):
        """A Bragg MTZ without an integer column cannot have its test set excluded."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        with pytest.raises(ValueError, match="No MTZInt column"):
            DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg_single,
                bragg_weight=1.0,
                exclude_free_reflections=True,
            )


# ---------------------------------------------------------------------------
# Separate reflection lists
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def thinned_diffuse(diffuse_bragg_targets, tmp_path_factory):
    """The diffuse target with every other reflection removed: ``(path, n_full, n_kept)``."""
    ds = rs.read_mtz(str(diffuse_bragg_targets.diffuse))
    thinned = ds.iloc[::2].copy()
    path = tmp_path_factory.mktemp("thinned") / "diffuse_thinned.mtz"
    thinned.write_mtz(str(path))
    return path, len(ds), len(thinned)


class TestSeparateReflectionLists:
    """Bragg and diffuse targets need not be indexed on the same reflections."""

    def test_each_term_keeps_its_own_reflections(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, thinned_diffuse
    ):
        """A thinned diffuse list leaves the Bragg list whole and the truth recoverable."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        path, n_full, n_thinned = thinned_diffuse
        reward = DiffuseBraggRewardFunction(
            bragg_target=diffuse_bragg_targets.bragg, diffuse_target=path, bragg_weight=0.5
        )

        assert len(reward._bragg_hkl) == n_full
        assert len(reward._diffuse_hkl) == n_thinned
        assert len(reward._hkl) == n_full

        atom_array, coords = diffuse_bragg_ensemble
        reward.prepare(RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu")
        loss, _ = score(reward, atom_array, coords)
        assert loss.item() < 1e-6

    def test_disjoint_lists_are_evaluated_together(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, thinned_diffuse
    ):
        """Bragg on the odd reflections, diffuse on the even: the union is the full set."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        diffuse_path, n_full, n_even = thinned_diffuse
        odd_bragg = diffuse_path.parent / "bragg_odd.mtz"
        rs.read_mtz(str(diffuse_bragg_targets.bragg)).iloc[1::2].copy().write_mtz(str(odd_bragg))

        reward = DiffuseBraggRewardFunction(
            bragg_target=odd_bragg, diffuse_target=diffuse_path, bragg_weight=0.5
        )
        assert len(reward._bragg_hkl) == n_full - n_even
        assert len(reward._hkl) == n_full

        atom_array, coords = diffuse_bragg_ensemble
        reward.prepare(RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu")
        loss, _ = score(reward, atom_array, coords)
        assert loss.item() < 1e-6

    @staticmethod
    def diffuse_on_cell(targets: DiffuseBraggTargets, tmp_path, scale_a: float):
        """The diffuse target rewritten with its ``a`` axis scaled by ``scale_a``."""
        ds = rs.read_mtz(str(targets.diffuse))
        c = targets.cell
        ds.cell = gemmi.UnitCell(c.a * scale_a, c.b, c.c, c.alpha, c.beta, c.gamma)
        path = tmp_path / f"diffuse_a_x{scale_a}.mtz"
        ds.write_mtz(str(path))
        return path

    def test_diffuse_cell_must_be_a_supercell_of_the_bragg_cell(
        self, diffuse_bragg_targets, tmp_path
    ):
        """A diffuse cell that is not an integer multiple of the Bragg cell is refused."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        other = self.diffuse_on_cell(diffuse_bragg_targets, tmp_path, 1.3)
        with pytest.raises(ValueError, match="integer supercell"):
            DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg, diffuse_target=other
            )

    def test_supercell_indexed_diffuse_is_not_yet_supported(self, diffuse_bragg_targets, tmp_path):
        """A diffuse target at fractional hkl is a valid input the reward cannot evaluate yet."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        doubled = self.diffuse_on_cell(diffuse_bragg_targets, tmp_path, 2.0)
        with pytest.raises(NotImplementedError, match=r"\(2, 1, 1\) supercell"):
            DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg, diffuse_target=doubled
            )


# ---------------------------------------------------------------------------
# Bulk solvent
# ---------------------------------------------------------------------------


class TestBulkSolvent:
    """The per-conformer solvent mask and how its cutoff is set."""

    def test_unknown_mode_is_rejected(self, diffuse_bragg_targets):
        """lunus has no single-mask mode, so ``"combined"`` is refused at construction."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        with pytest.raises(ValueError, match="bulk_solvent must be one of"):
            DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg, bragg_weight=0.5, bulk_solvent="combined"
            )

    def test_default_cutoff_is_calibrated_in_prepare(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, caplog
    ):
        """With no explicit cutoff, prepare() calibrates one and sets the taper to half of it."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        atom_array, _ = diffuse_bragg_ensemble
        reward = DiffuseBraggRewardFunction(
            bragg_target=diffuse_bragg_targets.bragg, bragg_weight=1.0, bulk_solvent="per_conformer"
        )
        assert reward._solvent is None  # nothing to calibrate against until prepare()

        reward.prepare(RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu")

        assert reward._solvent is not None and reward._solvent.cutoff > 0
        assert reward._solvent.taper_width == pytest.approx(0.5 * reward._solvent.cutoff)
        assert "Solvent cutoff calibrated" in caplog.text

    @pytest.mark.parametrize("chain", ["A", "B"])
    def test_recalibration_matches_a_fresh_prepare(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, chain
    ):
        """Re-preparing on a new topology gives the cutoff a fresh reward would, not a stale one."""
        atom_array, _ = diffuse_bragg_ensemble
        subset = atom_array[np.asarray(atom_array.chain_id) == chain]

        reward = prepared_reward(
            diffuse_bragg_targets, atom_array, 1.0, bulk_solvent="per_conformer"
        )
        full_cutoff = reward._solvent.cutoff
        reward.prepare(RewardInputs.from_atom_array(subset, ensemble_size=1), device="cpu")
        fresh = prepared_reward(diffuse_bragg_targets, subset, 1.0, bulk_solvent="per_conformer")

        assert reward._solvent.cutoff == pytest.approx(fresh._solvent.cutoff, rel=1e-6)
        assert reward._solvent.cutoff != pytest.approx(full_cutoff, rel=1e-3)

    @pytest.mark.parametrize("cutoff", [0.2, 0.35])
    def test_explicit_cutoff_is_used_as_given(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, cutoff
    ):
        """An explicit cutoff bypasses calibration and takes ``solvent_taper_width``."""
        from sampleworks.core.rewards.diffuse_bragg import DEFAULT_SOLVENT_TAPER_WIDTH

        atom_array, _ = diffuse_bragg_ensemble
        reward = prepared_reward(
            diffuse_bragg_targets,
            atom_array,
            1.0,
            bulk_solvent="per_conformer",
            solvent_cutoff=cutoff,
        )

        assert reward._solvent.cutoff == cutoff
        assert reward._solvent.taper_width == DEFAULT_SOLVENT_TAPER_WIDTH

    def test_per_conformer_solvent_changes_the_bragg_term(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, diffuse_bragg_rewards
    ):
        """A solvent contribution changes |Ftotal| relative to |Fprotein| under gradient."""
        atom_array, coords = diffuse_bragg_ensemble
        with_solvent = prepared_reward(
            diffuse_bragg_targets, atom_array, 1.0, bulk_solvent="per_conformer"
        )

        loss_off, _ = score(diffuse_bragg_rewards[1.0], atom_array, coords)
        loss_on, x = score(with_solvent, atom_array, coords, requires_grad=True)
        loss_on.backward()

        assert torch.isfinite(loss_on)
        assert not torch.isclose(loss_on, loss_off)
        assert x.grad is not None and torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
    def test_recovers_solvent_targets_with_the_calibrated_cutoff(
        self, diffuse_bragg_ensemble, diffuse_bragg_targets, weight
    ):
        """Targets generated with bulk solvent are recovered by the reward's own calibration.

        The generator and the reward calibrate the cutoff independently, each from
        gemmi's geometric solvent fraction on the first configuration, so a ~0 loss
        here shows the two agree, and the ``bulk_solvent="off"`` reward fits the
        same targets far worse. The analogue of the SF reward's Ftotal round trip.
        """
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        atom_array, coords = diffuse_bragg_ensemble
        inputs = RewardInputs.from_atom_array(atom_array, ensemble_size=1)

        def loss(bulk_solvent: str) -> float:
            reward = DiffuseBraggRewardFunction(
                bragg_target=diffuse_bragg_targets.bragg_solvent,
                diffuse_target=diffuse_bragg_targets.diffuse_solvent,
                bragg_weight=weight,
                bulk_solvent=bulk_solvent,
            )
            reward.prepare(inputs, device="cpu")
            return score(reward, atom_array, coords)[0].item()

        with_solvent, without = loss("per_conformer"), loss("off")
        assert with_solvent < 1e-6
        assert without > 1e3 * max(with_solvent, 1e-12)


# ---------------------------------------------------------------------------
# Registry builder and the single-configuration warning
# ---------------------------------------------------------------------------


class TestBuilder:
    """The registry builder: options in, a reward out, with the target checks it owns."""

    @staticmethod
    def build(targets: DiffuseBraggTargets, **overrides):
        """Build through the registry from both ensemble targets plus ``overrides``."""
        from sampleworks.core.rewards.diffuse_bragg import build_diffuse_bragg_reward
        from sampleworks.core.rewards.options import DiffuseBraggOptions
        from sampleworks.core.rewards.registry import RewardBuildContext

        options = {"bragg_target": str(targets.bragg), "diffuse_target": str(targets.diffuse)}
        options.update(overrides)
        return build_diffuse_bragg_reward(
            DiffuseBraggOptions(**options), RewardBuildContext(structure={})
        )

    def test_builds_from_options(self, diffuse_bragg_targets):
        """The options dataclass reaches the reward's constructor."""
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        reward = self.build(diffuse_bragg_targets, bragg_weight=0.3)

        assert isinstance(reward, DiffuseBraggRewardFunction)
        assert reward.bragg_weight == 0.3

    def test_is_registered(self):
        """The registry knows the reward under its CLI name."""
        from sampleworks.core.rewards.registry import get_reward_spec

        assert get_reward_spec("diffuse_bragg").options_cls.__name__ == "DiffuseBraggOptions"

    def test_weighted_bragg_term_requires_its_target(self, diffuse_bragg_targets):
        """The builder's own check names the CLI flag."""
        with pytest.raises(ValueError, match="--bragg-weight is 0"):
            self.build(diffuse_bragg_targets, bragg_target=None)

    def test_weighted_diffuse_term_requires_its_target(self, diffuse_bragg_targets):
        """The builder's own check names the CLI flag."""
        with pytest.raises(ValueError, match="--bragg-weight is 1"):
            self.build(diffuse_bragg_targets, diffuse_target=None)

    def test_pure_diffuse_needs_no_bragg_target(self, diffuse_bragg_targets):
        """A zero Bragg weight builds without a Bragg target."""
        reward = self.build(diffuse_bragg_targets, bragg_target=None, bragg_weight=0.0)

        assert reward.bragg_weight == 0.0

    def test_weight_outside_the_unit_interval_is_rejected(self, diffuse_bragg_targets):
        """The builder rejects a non-convex weight before constructing the reward."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            self.build(diffuse_bragg_targets, bragg_weight=1.5)


def test_single_configuration_with_a_weighted_diffuse_term_warns(
    diffuse_bragg_ensemble, diffuse_bragg_targets, caplog
):
    """The diffuse term of one configuration is identically zero, so prepare() says so."""
    atom_array, _ = diffuse_bragg_ensemble
    prepared_reward(diffuse_bragg_targets, atom_array, 0.5)

    assert "identically zero" in caplog.text


def test_pure_bragg_single_configuration_does_not_warn(
    diffuse_bragg_ensemble, diffuse_bragg_targets, caplog
):
    """A pure-Bragg reward has no diffuse term to warn about."""
    atom_array, _ = diffuse_bragg_ensemble
    prepared_reward(diffuse_bragg_targets, atom_array, 1.0)

    assert "identically zero" not in caplog.text
