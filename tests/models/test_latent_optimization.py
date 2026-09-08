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
from sampleworks.core.scalers.latent_optimization import LatentAnchor, LatentOptimization
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
        torch.testing.assert_close(leaf.detach(), base)  # leaf starts exactly at the baseline
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
