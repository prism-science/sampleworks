"""Tests for the two-phase reward protocol and its dispatch helper.

Reciprocal-space rewards cannot be built from configuration alone: they need the
atom array the sampled coordinates correspond to, which only exists once
sampling has started. ``prepare_reward_if_needed`` is the one place that decides
*which* array that is, and both trajectory scalers call it. Getting the choice
wrong is silent -- the model and structure arrays can have the same length while
ordering atoms differently -- so it is worth pinning directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
)
from sampleworks.eval.structure_utils import SampleworksProcessedStructure
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.atom_reconciler import AtomReconciler


def _atom_array(n_atoms: int, element: str = "C") -> AtomArray:
    """A minimal but valid AtomArray: one chain, one residue per atom."""
    array = AtomArray(n_atoms)
    array.coord = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3)
    array.chain_id = np.full(n_atoms, "A")
    array.res_id = np.arange(1, n_atoms + 1)
    array.res_name = np.full(n_atoms, "GLY")
    array.atom_name = np.full(n_atoms, "CA")
    array.element = np.full(n_atoms, element)
    array.b_factor = np.full(n_atoms, 20.0, dtype=np.float32)
    array.occupancy = np.ones(n_atoms, dtype=np.float32)
    return array


def _processed(atom_array: AtomArray, model_atom_array: AtomArray | None = None):
    """A SampleworksProcessedStructure carrying the two arrays under test."""
    reference = model_atom_array if model_atom_array is not None else atom_array
    return SampleworksProcessedStructure(
        structure={"asym_unit": atom_array},
        model_input=GenerativeModelInput(conditioning=None),
        input_coords=torch.from_numpy(reference.coord).unsqueeze(0),
        atom_array=atom_array,
        ensemble_size=1,
        reconciler=AtomReconciler.from_arrays(reference, atom_array),
        model_atom_array=model_atom_array,
    )


class _PreparableReward:
    """Records what it was prepared with. Satisfies the preparable protocol."""

    def __init__(self) -> None:
        self.prepared_with: list[tuple[AtomArray, str]] = []

    def __call__(self, coordinates, **kwargs):
        return torch.zeros(coordinates.shape[0])

    def prepare(self, atom_array: AtomArray, *, device: torch.device | str = "cpu") -> None:
        self.prepared_with.append((atom_array, str(device)))


class _PlainReward:
    """A reward with no prepare(); must be left alone."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, coordinates, **kwargs):
        self.calls += 1
        return torch.zeros(coordinates.shape[0])


class TestPreparableProtocol:
    """The protocol must discriminate on prepare() alone, at runtime."""

    def test_reward_with_prepare_satisfies_the_protocol(self):
        assert isinstance(_PreparableReward(), PreparableRewardFunctionProtocol)

    def test_reward_without_prepare_does_not(self):
        reward = _PlainReward()
        assert not isinstance(reward, PreparableRewardFunctionProtocol)
        # ...but is still a reward, so the narrower check is the only difference.
        assert isinstance(reward, RewardFunctionProtocol)


class TestPrepareRewardIfNeeded:
    """Which array the reward is prepared with, and when it is skipped."""

    def test_prepares_with_the_model_array_when_one_exists(self):
        """The sampled coordinates follow the model's topology, not the input file's.

        Both arrays here have the same length and differ only in element, which
        is exactly the case no shape check would catch.
        """
        structure_array = _atom_array(4, element="C")
        model_array = _atom_array(4, element="N")

        reward = _PreparableReward()
        prepare_reward_if_needed(reward, _processed(structure_array, model_array))

        assert len(reward.prepared_with) == 1
        prepared_array, _ = reward.prepared_with[0]
        assert prepared_array is model_array
        assert set(np.asarray(prepared_array.element)) == {"N"}

    def test_falls_back_to_the_structure_array_when_the_model_has_none(self):
        """model_atom_array is None when the model shares the input topology."""
        structure_array = _atom_array(4)

        reward = _PreparableReward()
        prepare_reward_if_needed(reward, _processed(structure_array, None))

        assert len(reward.prepared_with) == 1
        assert reward.prepared_with[0][0] is structure_array

    def test_forwards_the_device(self):
        reward = _PreparableReward()
        prepare_reward_if_needed(reward, _processed(_atom_array(2)), device=torch.device("cpu"))

        assert reward.prepared_with[0][1] == "cpu"

    def test_leaves_a_reward_without_prepare_untouched(self):
        """No-op rather than an error: most rewards need nothing extra."""
        reward = _PlainReward()
        prepare_reward_if_needed(reward, _processed(_atom_array(2)))

        assert reward.calls == 0

    def test_accepts_no_reward_at_all(self):
        """Unguided sampling passes None; the scalers call this unconditionally."""
        prepare_reward_if_needed(None, _processed(_atom_array(2)))

    def test_prepares_once_per_call_and_not_on_construction(self):
        """The two-phase contract: __init__ takes config, prepare takes the array."""
        reward = _PreparableReward()
        assert reward.prepared_with == []

        processed = _processed(_atom_array(3))
        prepare_reward_if_needed(reward, processed)
        assert len(reward.prepared_with) == 1


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_device_accepts_both_string_and_torch_device(device):
    """Callers pass coords.device; the signature also documents a string."""
    reward = _PreparableReward()
    prepare_reward_if_needed(reward, _processed(_atom_array(2)), device=device)

    assert reward.prepared_with[0][1] == "cpu"
