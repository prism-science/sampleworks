"""Which atom array the reward tensors and the prepare hook both follow.

``SampleworksProcessedStructure.reward_atom_array`` is the single expression of
a choice two callers depend on: ``to_reward_inputs`` builds the reward tensors
from it, and the trajectory scalers pass it to ``prepare_reward_if_needed``. If
those two ever disagreed, a reward would be prepared for one atom ordering and
scored on another — and no shape check would catch it, because the model and
structure arrays can have the same length. The property exists so there is only
one answer; these pin what it is.
"""

from __future__ import annotations

import numpy as np
import torch
from biotite.structure import AtomArray
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.structure_utils import SampleworksProcessedStructure


def _atom_array(n_atoms: int, element: str = "C") -> AtomArray:
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


def test_prefers_the_model_array_when_the_model_exposes_one():
    """The sampled coordinates follow the model's topology, not the input file's.

    Both arrays here are the same length and differ only in element, which is
    the case no shape check would catch.
    """
    structure_array = _atom_array(4, element="C")
    model_array = _atom_array(4, element="N")

    processed = _processed(structure_array, model_array)

    assert processed.reward_atom_array is model_array


def test_falls_back_to_the_structure_array_when_the_model_has_none():
    """model_atom_array is None when the model shares the input topology."""
    structure_array = _atom_array(4)

    processed = _processed(structure_array, None)

    assert processed.reward_atom_array is structure_array


def test_reward_inputs_are_built_from_the_same_array():
    """The whole point of the property: one choice, not two that can drift."""
    structure_array = _atom_array(4, element="C")
    model_array = _atom_array(4, element="N")
    processed = _processed(structure_array, model_array)

    reward_inputs = processed.to_reward_inputs()

    assert reward_inputs.elements.shape[-1] == len(processed.reward_atom_array)
