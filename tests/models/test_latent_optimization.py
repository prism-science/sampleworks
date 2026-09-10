"""Unit tests for the IT-opt scaler's core pieces.

Covers the two genuinely-new, model-free parts of ``LatentOptimization``:

- :class:`LatentAnchor` -- the on-manifold prior ``Σ w_i·mean((latent_i − baseline_i)²)``.
- :meth:`LatentOptimization._leaf_latents` -- turning the cached ``s``/``z`` on the
  conditioning into fresh optimizable leaves (detach → clone → ``requires_grad``),
  and raising loudly when the configured attribute names match no latent.

CPU-only, no model checkpoints: a minimal frozen-dataclass conditioning stands in
for the real wrappers' ``@dataclass(frozen=True, slots=True)`` conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from sampleworks.core.scalers.latent_optimization import (
    _PerMemberStepper,
    LatentAnchor,
    LatentOptimization,
)
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import GenerativeModelInput
from torch import Tensor


@dataclass(frozen=True)
class _Cond:
    """Minimal conditioning carrying a single ``s`` and pair ``z`` (like the wrappers)."""

    s: Tensor
    z: Tensor


@pytest.fixture
def features() -> GenerativeModelInput:
    torch.manual_seed(0)
    cond = _Cond(s=torch.randn(1, 6, 4), z=torch.randn(1, 6, 6, 2))
    return GenerativeModelInput(conditioning=cond)


def _make_opt(**overrides) -> LatentOptimization:
    """Build a scaler for the leaf-logic tests. ``__init__`` only stores attrs + logs,
    so no model/checkpoint is needed."""
    kwargs = dict(
        optimize_single=True,
        optimize_pair=True,
        single_attr="s",
        pair_attr="z",
        anchor_weight_single=0.1,
        anchor_weight_pair=0.2,
    )
    kwargs.update(overrides)
    return LatentOptimization(**kwargs)


# --- LatentAnchor -------------------------------------------------------------


def test_anchor_is_zero_at_the_baseline():
    torch.manual_seed(0)
    s, z = torch.randn(1, 6, 4), torch.randn(1, 6, 6, 2)
    # latent == baseline -> no drift -> zero penalty, regardless of the weights
    assert LatentAnchor([1.0, 2.0])([s, z], [s, z]).item() == pytest.approx(0.0)


def test_anchor_is_weighted_mean_squared_deviation():
    s0, z0 = torch.zeros(4), torch.zeros(4)
    s, z = torch.full((4,), 2.0), torch.full((4,), 3.0)  # mean sq dev = 4 and 9
    # 0.5 * 4 + 2.0 * 9 = 20
    assert LatentAnchor([0.5, 2.0])([s, z], [s0, z0]).item() == pytest.approx(20.0)


def test_anchor_gradient_reaches_the_latent():
    base = torch.zeros(4)
    leaf = torch.full((4,), 2.0).requires_grad_(True)
    LatentAnchor([1.0])([leaf], [base]).backward()
    assert leaf.grad is not None
    # d/dx of mean(x²) is 2x/n = 2*2/4 = 1 per element
    torch.testing.assert_close(leaf.grad, torch.full((4,), 1.0))


# --- LatentOptimization._leaf_latents -----------------------------------------


def test_leaf_latents_makes_requires_grad_leaves(features):
    opt = _make_opt()
    io = AttrLatentIO("s", "z")
    new_features, latents, baselines, weights = opt._leaf_latents(features, io)

    assert len(latents) == 2
    for leaf in latents:
        assert leaf.requires_grad and leaf.is_leaf  # a true leaf Adam can update directly
    for leaf, base in zip(latents, baselines):
        assert not base.requires_grad  # the anchor target is detached
        # one leaf per ensemble member on a leading batch dim; the baseline stays un-batched
        for member in leaf.detach():
            torch.testing.assert_close(member, base)  # every member starts at the baseline
    # the rewritten conditioning holds the SAME leaf objects, so the model reads
    # the optimizable tensor rather than the original cached one
    assert new_features.conditioning.s is latents[0]
    assert new_features.conditioning.z is latents[1]
    assert weights == [0.1, 0.2]


def test_leaf_latents_are_severed_from_the_original(features):
    opt = _make_opt()
    _, latents, baselines, _ = opt._leaf_latents(features, AttrLatentIO("s", "z"))
    # a fresh clone/detach, not the original cached tensor (so optimizing it never
    # writes back into, or backprops through, the frozen trunk output)
    assert latents[0] is not features.conditioning.s
    assert baselines[0] is not features.conditioning.s


def test_leaf_latents_single_only_respects_the_flag(features):
    opt = _make_opt(optimize_pair=False)
    _, latents, _, weights = opt._leaf_latents(features, AttrLatentIO("s", "z"))
    assert len(latents) == 1  # pair skipped even though the io could address it
    assert weights == [0.1]


def test_leaf_latents_raises_when_no_attribute_matches(features):
    # wrong attribute names -> nothing to optimize -> a clear error, not a silent no-op
    opt = _make_opt()
    io = AttrLatentIO("does_not_exist", "also_missing")
    with pytest.raises(ValueError, match="does not expose"):
        opt._leaf_latents(features, io)


def test_leaf_latents_raises_when_only_one_attribute_matches(features):
    # The dangerous case: one name is right, so the run would otherwise optimize half of what
    # was asked for and still look successful.
    opt = _make_opt()
    io = AttrLatentIO("does_not_exist", "z")
    with pytest.raises(ValueError, match="does not expose"):
        opt._leaf_latents(features, io)


def test_leaf_latents_raises_when_nothing_is_enabled(features):
    opt = _make_opt(optimize_single=False, optimize_pair=False)
    with pytest.raises(ValueError, match="no latent enabled"):
        opt._leaf_latents(features, AttrLatentIO("s", "z"))


# --- _PerMemberStepper --------------------------------------------------------
#
# _PerMemberStepper exists because each ensemble member now owns a separate latent, so ``s`` and
# ``z`` carry a leading ensemble dimension that a stock diffusion module will not accept. The
# stepper slices that dimension apart and calls the wrapped model once per member. The tests below
# check the four things that slicing has to get right: every member is denoised with its own
# latent, a latent that is not being optimized stays shared, a per-member timestep is sliced too,
# and the members' gradients stay separate.


class _RecordingModel:
    """Stand-in wrapper that records the arguments of every ``step`` call.

    ``step`` hands ``x_t`` straight back, so the stacked result is the input reassembled and any
    mis-ordering of the members shows up directly in the returned tensor.
    """

    def __init__(self):
        self.calls = []  # one (x_t, t, s, z) tuple per member, in call order

    def step(self, x_t, t, *, features):
        cond = features.conditioning
        self.calls.append((x_t, t, cond.s, cond.z))
        return x_t


class _ScalingModel:
    """Stand-in wrapper that scales a member's coordinates by that member's own latents.

    The gradient test needs a model whose output actually depends on the latents, so that
    ``backward`` sends a gradient back to each member's leaf. Multiplying by the sums keeps the
    dependence obvious and the shapes trivial.
    """

    def step(self, x_t, t, *, features):
        cond = features.conditioning
        return x_t * cond.s.sum() + x_t * cond.z.sum()


def _per_member(shape, n_members):
    """Stack ``n_members`` tensors of ``shape``, filling member i with the number i.

    Giving each member a distinct constant means a mis-sliced member is visible in an assertion
    rather than hidden behind values that happen to match.
    """
    return torch.stack([torch.full(shape, float(i)) for i in range(n_members)])


def _stepper(model, *, optimize_single=True, optimize_pair=True):
    """Build a stepper over ``model`` for the ``s``/``z`` names the fake conditioning uses."""
    return _PerMemberStepper(
        model,
        AttrLatentIO("s", "z"),
        optimize_single=optimize_single,
        optimize_pair=optimize_pair,
    )


def test_per_member_stepper_gives_each_member_its_own_latent():
    n_members = 3
    s = _per_member((6, 4), n_members)
    z = _per_member((6, 6, 2), n_members)
    x_t = _per_member((5, 3), n_members)
    features = GenerativeModelInput(conditioning=_Cond(s=s, z=z))

    model = _RecordingModel()
    output = _stepper(model).step(x_t, torch.tensor(0.5), features=features)

    # One un-batched forward per member rather than a single batched call.
    assert len(model.calls) == n_members
    for i, (x_i, _, s_i, z_i) in enumerate(model.calls):
        torch.testing.assert_close(s_i, s[i])  # member i's own latent, ensemble dim stripped
        torch.testing.assert_close(z_i, z[i])
        torch.testing.assert_close(x_i, x_t[i : i + 1])  # and only member i's coordinates
    torch.testing.assert_close(output, x_t)  # members stacked back in their original order


def test_per_member_stepper_leaves_an_unoptimized_latent_shared():
    # Only an optimized latent carries the ensemble dimension. A latent that is not being
    # optimized is still the shared un-batched baseline, so slicing it would take the wrong tensor.
    n_members = 2
    shared_s = torch.randn(6, 4)
    z = _per_member((6, 6, 2), n_members)
    features = GenerativeModelInput(conditioning=_Cond(s=shared_s, z=z))

    model = _RecordingModel()
    x_t = _per_member((5, 3), n_members)
    _stepper(model, optimize_single=False).step(
        x_t, torch.tensor(0.5), features=features
    )

    for i, (_, _, s_i, z_i) in enumerate(model.calls):
        torch.testing.assert_close(s_i, shared_s)  # the same shared latent for every member
        torch.testing.assert_close(z_i, z[i])


def test_per_member_stepper_slices_a_per_member_timestep():
    # A schedule may hand out one timestep per member, in which case each member must receive its
    # own value rather than the whole batch.
    n_members = 3
    features = GenerativeModelInput(
        conditioning=_Cond(s=_per_member((6, 4), n_members), z=_per_member((6, 6, 2), n_members))
    )
    t = torch.tensor([0.1, 0.2, 0.3])

    model = _RecordingModel()
    _stepper(model).step(_per_member((5, 3), n_members), t, features=features)

    for i, (_, t_i, _, _) in enumerate(model.calls):
        torch.testing.assert_close(t_i, t[i : i + 1])


def test_per_member_stepper_keeps_member_gradients_separate():
    # This is the point of giving every member its own leaf: a member's gradient must depend on
    # that member alone. If the members shared one latent, every slice would receive the same
    # gradient and the ensemble would collapse back onto a single solution.
    n_members = 3
    opt = _make_opt(ensemble_size=n_members)
    features = GenerativeModelInput(
        conditioning=_Cond(s=torch.randn(1, 6, 4), z=torch.randn(1, 6, 6, 2))
    )
    leaf_features, latents, _, _ = opt._leaf_latents(features, AttrLatentIO("s", "z"))

    # Each member gets different coordinates, so a correctly sliced gradient differs per member.
    x_t = _per_member((5, 3), n_members) + 1.0
    stepper = _stepper(_ScalingModel())
    output = stepper.step(x_t, torch.tensor(0.5), features=leaf_features)
    output.sum().backward()

    for leaf in latents:
        assert leaf.grad.shape == leaf.shape
        for i in range(n_members):
            assert leaf.grad[i].abs().sum() > 0  # every member actually received a gradient
        # Members must not all share one gradient, which is what a single shared latent would give.
        assert not torch.allclose(leaf.grad[0], leaf.grad[1])
