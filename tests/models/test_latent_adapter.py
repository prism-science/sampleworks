"""Tests for the latent read/write seam (``AttrLatentIO``) used by IT-opt.

These run on CPU with no model checkpoints, using minimal mock conditioning
dataclasses whose representations are stored in named attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from sampleworks.models.latent_adapter import AttrLatentIO, LatentIO
from torch import Tensor


@dataclass(frozen=True)
class _SingleRepConditioning:
    """Minimal conditioning carrying a single representation ``s`` plus sidecar state."""

    s: Tensor
    sidecar: str = "untouched"  # stands in for non-tensor state (e.g. RF3 chiral arrays)


@dataclass(frozen=True)
class _PairConditioning:
    """Conditioning carrying both single ``s`` and pair ``z`` representations."""

    s: Tensor
    z: Tensor
    sidecar: str = "untouched"


@pytest.fixture
def s_tensor() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 8, 4)  # [batch, tokens, d_s]


@pytest.fixture
def z_tensor() -> Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 8, 8, 2)  # [batch, tokens, tokens, d_z]


# --- AttrLatentIO: the only model-specific knowledge -------------------------


def test_attr_latent_io_roundtrip(s_tensor: Tensor):
    io = AttrLatentIO(single_attr="s")
    cond = _SingleRepConditioning(s=s_tensor)
    assert io.read_single(cond) is s_tensor
    new = io.write_single(cond, s_tensor * 5)
    torch.testing.assert_close(new.s, s_tensor * 5)


def test_attr_latent_io_preserves_sidecar_state(s_tensor: Tensor):
    """replace() must leave non-tensor state (e.g. RF3 chiral arrays) intact."""
    io = AttrLatentIO(single_attr="s")
    cond = _SingleRepConditioning(s=s_tensor, sidecar="keep-me")
    new = io.write_single(cond, s_tensor + 1)
    assert new.sidecar == "keep-me"


def test_attr_latent_io_satisfies_protocol():
    assert isinstance(AttrLatentIO("s"), LatentIO)


def test_attr_latent_io_missing_attr_raises(s_tensor: Tensor):
    """A name that matches no attribute is a configuration error, not an empty read."""
    io = AttrLatentIO(single_attr="does_not_exist")
    with pytest.raises(AttributeError):
        io.read_single(_SingleRepConditioning(s=s_tensor))


# --- Pair (z) representation support -----------------------------------------


def test_pair_io_none_by_default_is_noop(s_tensor: Tensor, z_tensor: Tensor):
    """Single-arg AttrLatentIO leaves the pair rep unreachable (backward compatible)."""
    io = AttrLatentIO("s")
    cond = _PairConditioning(s=s_tensor, z=z_tensor)
    assert io.read_pair(cond) is None
    assert io.write_pair(cond, z_tensor * 9) is cond  # unchanged


def test_pair_io_roundtrip(s_tensor: Tensor, z_tensor: Tensor):
    io = AttrLatentIO(single_attr="s", pair_attr="z")
    cond = _PairConditioning(s=s_tensor, z=z_tensor)
    assert io.read_pair(cond) is z_tensor
    new = io.write_pair(cond, z_tensor * 3)
    torch.testing.assert_close(new.z, z_tensor * 3)
    torch.testing.assert_close(new.s, s_tensor)  # single untouched
    assert new.sidecar == "untouched"
