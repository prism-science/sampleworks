"""Behavioural tests for the combined Bragg + diffuse reward.

Three properties, in increasing order of what they actually establish:

**Recovery.** The loss is ~0 when the model is the structure the targets were
generated from. Necessary but weak — a reward that returned a constant would
also pass in isolation.

**Discrimination.** The loss rises monotonically as the model is displaced from
the truth. This is what a reward has to do to be worth anything, and it is where
a term dominated by its own numerical noise fails.

**Gradients.** ``backward()`` reaches the coordinates with finite, non-zero
values, and points downhill. Guidance uses nothing but this.

The targets are generated in-process rather than read from a checked-in file, so
the fixtures cannot drift from the generator that made them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
    RewardInputs,
)


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Slow, but needing neither a GPU nor model weights: ~3 min, dominated by ~30
# forward passes on CPU. Marked at module scope, following
# tests/eval/test_rscc_grid_search_script.py, where `slow` already covers
# runtime alone rather than hardware.
pytestmark = pytest.mark.slow

RESOLUTION = 1.8
SOURCE_CIF = "1vme_final.cif"


@pytest.fixture(scope="module")
def ensemble(resources_dir: Path):
    """1VME expanded into one configuration per altloc, with its topology."""
    from sampleworks.synthetic.generate_synthetic_sf import BatchRowForMTZ
    from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations

    source = resources_dir / "1vme" / SOURCE_CIF
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")

    atom_array, coords = load_configurations(
        source, BatchRowForMTZ(filename=SOURCE_CIF), "default", altlocs_as_models=True
    )
    assert coords.shape[0] >= 2, "1VME should expand to at least two conformations"
    return atom_array, coords


@pytest.fixture(scope="module")
def targets(ensemble, resources_dir: Path, tmp_path_factory):
    """Bragg and diffuse MTZs generated from the ensemble itself.

    Generating rather than checking in means the target and the thing being
    scored cannot drift apart, and the recovery test below is then a genuine
    round trip through both writers and the reader.
    """
    import gemmi
    from sampleworks.synthetic.generate_synthetic_sf_lunus import (
        compute_ensemble_amplitudes,
        dataset_from_bragg_amplitudes,
        dataset_from_diffuse_intensities,
        save_mtz,
    )

    atom_array, coords = ensemble
    meta = gemmi.read_structure(str(resources_dir / "1vme" / SOURCE_CIF))
    cell, spacegroup = meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)

    hkl, mean_f, diffuse = compute_ensemble_amplitudes(
        atom_array, coords, cell, spacegroup, RESOLUTION, torch.device("cpu")
    )

    out = tmp_path_factory.mktemp("diffuse_bragg_targets")
    bragg_path, diffuse_path = out / "bragg.mtz", out / "diffuse.mtz"
    save_mtz(
        dataset_from_bragg_amplitudes(hkl, mean_f, cell, spacegroup, test_fraction=0.0),
        bragg_path,
        "structure factors",
    )
    save_mtz(
        dataset_from_diffuse_intensities(hkl, diffuse, cell, spacegroup),
        diffuse_path,
        "diffuse intensities",
    )

    assert diffuse.max() > 0, "a two-state ensemble must have nonzero diffuse"
    return bragg_path, diffuse_path, cell, spacegroup


def build_reward(targets, weight, **kwargs):
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

    bragg_path, diffuse_path, _, _ = targets
    return DiffuseBraggRewardFunction(
        bragg_target=bragg_path, diffuse_target=diffuse_path, bragg_weight=weight, **kwargs
    )


def prepare(reward, atom_array, device="cpu"):
    """Prepare the way the sampling loop does, through RewardInputs.

    ``prepare()`` takes reward inputs rather than an atom array, so the topology
    is wrapped as single-conformer inputs -- the same shape
    ``StructureFactorRewardFunction``'s tests use.
    """
    reward.prepare(
        RewardInputs.from_atom_array(atom_array, ensemble_size=1, device=device), device=device
    )
    return reward


def score(reward, atom_array, coords, requires_grad=False):
    """Evaluate the reward the way the sampling loop would.

    Occupancies are divided by the configuration count because the reward undoes
    the ``1/N`` convention ``RewardInputs`` imposes; passing them straight
    through would double-weight every atom.
    """
    n = coords.shape[0]
    x = torch.as_tensor(coords, dtype=torch.float32)
    if requires_grad:
        x = x.detach().clone().requires_grad_(True)
    occ = torch.as_tensor(np.tile(np.asarray(atom_array.occupancy, dtype=np.float32) / n, (n, 1)))
    dummy = torch.zeros_like(occ)
    return reward(x, dummy.long(), dummy, occ), x


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_recovers_the_structure_it_was_generated_from(ensemble, targets, weight):
    """~0 loss when the model is the truth, at every mixture.

    Measured on 1VME (2 conformations, 86499 reflections): 2e-14 pure Bragg,
    6e-10 pure diffuse. Bragg reaches float32 epsilon because that is what the
    MTZ stores; diffuse is looser because the anisotropic component is what
    survives subtracting a large radial background, which amplifies the relative
    error of the intensities it came from.
    """
    atom_array, coords = ensemble
    reward = build_reward(targets, weight)
    prepare(reward, atom_array)

    loss, _ = score(reward, atom_array, coords)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_loss_increases_with_displacement(ensemble, targets, weight):
    """Monotonic in how far the model has been moved from the truth.

    The displacement is a fixed random direction scaled up, so the sequence
    probes one path away from the minimum rather than several unrelated points.
    A term dominated by its own numerical noise would fail here while still
    passing the recovery test above.
    """
    atom_array, coords = ensemble
    reward = build_reward(targets, weight)
    prepare(reward, atom_array)

    rng = np.random.default_rng(0)
    direction = rng.normal(size=coords.shape)
    losses = [
        score(reward, atom_array, coords + amplitude * direction)[0].item()
        for amplitude in (0.0, 0.02, 0.05, 0.1, 0.2)
    ]

    assert all(np.isfinite(losses))
    assert losses == sorted(losses), f"loss is not monotonic in displacement: {losses}"
    assert losses[-1] > 1e3 * max(losses[0], 1e-12), (
        f"a 0.2 A displacement barely moved the loss: {losses}"
    )


@pytest.mark.parametrize("weight", [1.0, 0.0, 0.5])
def test_gradients_reach_the_coordinates(ensemble, targets, weight):
    """backward() puts finite, non-zero gradients on every configuration.

    Guidance consumes nothing but this. Gradients are checked per configuration
    because an ensemble axis that collapsed -- all members receiving the same
    gradient, or only the first receiving any -- would still produce a plausible
    scalar loss.
    """
    atom_array, coords = ensemble
    reward = build_reward(targets, weight)
    prepare(reward, atom_array)

    rng = np.random.default_rng(1)
    displaced = coords + 0.1 * rng.normal(size=coords.shape)
    loss, x = score(reward, atom_array, displaced, requires_grad=True)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for i in range(coords.shape[0]):
        assert x.grad[i].abs().max() > 0, f"configuration {i} received no gradient"


def test_gradient_points_downhill(ensemble, targets):
    """A small step along -grad lowers the loss.

    The sign convention matters and is invisible in the magnitude checks above:
    a reward whose gradient pointed uphill would look healthy in every other test
    here and drive guidance away from the data.
    """
    atom_array, coords = ensemble
    reward = build_reward(targets, 0.5)
    prepare(reward, atom_array)

    rng = np.random.default_rng(2)
    displaced = coords + 0.1 * rng.normal(size=coords.shape)
    loss, x = score(reward, atom_array, displaced, requires_grad=True)
    loss.backward()

    step = 1e-3 / x.grad.abs().max().item()
    stepped = displaced - step * x.grad.detach().numpy()
    lowered, _ = score(reward, atom_array, stepped)

    assert lowered.item() < loss.item(), (
        f"stepping along -grad raised the loss: {loss.item():.6e} -> {lowered.item():.6e}"
    )


class TestWeightHandling:
    """The convex weight, and what a zero weight is allowed to require."""

    def test_zero_weight_does_not_require_its_target(self, targets):
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        bragg_path, diffuse_path, _, _ = targets
        # Pure diffuse needs no Bragg target, and vice versa.
        DiffuseBraggRewardFunction(diffuse_target=diffuse_path, bragg_weight=0.0)
        DiffuseBraggRewardFunction(bragg_target=bragg_path, bragg_weight=1.0)

    def test_nonzero_weight_requires_its_target(self, targets):
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        bragg_path, diffuse_path, _, _ = targets
        with pytest.raises(ValueError, match="bragg_target is required"):
            DiffuseBraggRewardFunction(diffuse_target=diffuse_path, bragg_weight=0.5)
        with pytest.raises(ValueError, match="diffuse_target is required"):
            DiffuseBraggRewardFunction(bragg_target=bragg_path, bragg_weight=0.5)

    def test_weight_outside_the_unit_interval_is_rejected(self, targets):
        from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction

        bragg_path, diffuse_path, _, _ = targets
        with pytest.raises(ValueError, match="convex"):
            DiffuseBraggRewardFunction(
                bragg_target=bragg_path, diffuse_target=diffuse_path, bragg_weight=1.5
            )

    def test_callable_weight_without_a_time_is_a_clear_error(self, ensemble, targets):
        """RewardFunctionProtocol carries no t, so nothing sets current_time yet."""
        atom_array, coords = ensemble
        reward = build_reward(targets, lambda t: t)
        prepare(reward, atom_array)

        with pytest.raises(RuntimeError, match="current_time"):
            score(reward, atom_array, coords)

    def test_callable_weight_uses_current_time(self, ensemble, targets):
        atom_array, coords = ensemble
        reward = build_reward(targets, lambda t: 1.0 if t > 0.5 else 0.0)
        prepare(reward, atom_array)

        reward.current_time = 0.9
        pure_bragg, _ = score(reward, atom_array, coords)
        reward.current_time = 0.1
        pure_diffuse, _ = score(reward, atom_array, coords)

        # Both are recovery losses, but of different terms, so they differ by
        # the several orders of magnitude the recovery test documents.
        assert pure_bragg.item() != pure_diffuse.item()


class TestProtocolConformance:
    """The two-phase hook, exercised the way the sampling loop reaches it.

    ``isinstance`` is structural and checks method *names*, so it cannot catch a
    ``prepare`` whose first parameter is the wrong type: before #402's signature
    change was followed here, this class satisfied the protocol while
    ``prepare_reward_if_needed`` handed it a ``RewardInputs`` where an
    ``AtomArray`` was expected, and the failure surfaced deep inside
    ``build_setup``. Going through the hook is what pins it.
    """

    def test_satisfies_both_reward_protocols(self, targets):
        reward = build_reward(targets, 0.5)

        assert isinstance(reward, RewardFunctionProtocol)
        assert isinstance(reward, PreparableRewardFunctionProtocol)

    def test_prepare_hook_binds_the_reward(self, ensemble, targets):
        """``prepare_reward_if_needed`` is how scalers prepare a reward, so the
        reward has to be usable after being prepared that way and no other."""
        atom_array, coords = ensemble
        reward = build_reward(targets, 0.5)

        prepare_reward_if_needed(
            reward, RewardInputs.from_atom_array(atom_array, ensemble_size=1), device="cpu"
        )

        assert reward.setup is not None
        loss, _ = score(reward, atom_array, coords)
        assert torch.isfinite(loss)

    def test_prepare_without_an_atom_array_is_a_clear_error(self, ensemble, targets):
        """The topology cannot be recovered from the tensors alone."""
        atom_array, _ = ensemble
        reward = build_reward(targets, 0.5)
        inputs = RewardInputs.from_atom_array(atom_array, ensemble_size=1)
        inputs.atom_array = None

        with pytest.raises(ValueError, match="atom_array"):
            reward.prepare(inputs)
