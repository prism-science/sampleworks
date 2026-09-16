"""Tests for RewardInputs.

``from_atom_array`` must reject atom arrays with NaN B-factors, NaN coordinates, and
invalid occupancies, and keep the topology it was built from. ``to_atom_array`` must
rebuild the reference array a reward's ``prepare()`` needs without touching that
topology.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from biotite.structure import array, Atom, AtomArray, stack
from sampleworks.core.rewards.protocol import RewardInputs


class TestRewardInputsFromAtomArray:
    """Validate that RewardInputs.from_atom_array rejects invalid atom arrays."""

    def test_nan_b_factors_from_missing_atoms_rejected(self, atom_array_1vme_with_missing_atoms):
        """
        RewardInputs rejects the raw atom array from add_missing_atoms.
        """
        with pytest.raises(ValueError, match="NaN B-factors"):
            RewardInputs.from_atom_array(atom_array_1vme_with_missing_atoms, ensemble_size=1)

    def test_cleaned_missing_atoms_accepted(self, atom_array_1vme_with_missing_atoms):
        """After applying the same fixes as the RF3 wrapper, the atom array passes."""
        aa = atom_array_1vme_with_missing_atoms.copy()

        # Fix NaN coordinates
        nan_coord_mask = np.any(np.isnan(aa.coord), axis=-1)
        if nan_coord_mask.any():
            resolved_coords = aa.coord[~nan_coord_mask]
            if len(resolved_coords) > 0:
                centroid = resolved_coords.mean(axis=0)
            else:
                centroid = np.zeros(3)
            n_nan = int(nan_coord_mask.sum())
            noise = np.random.normal(loc=0.0, scale=1.0, size=(n_nan, 3)).astype(np.float32)
            new_coords = aa.coord.copy()
            new_coords[nan_coord_mask] = centroid + noise
            aa.coord = new_coords

        # Fix occupancy
        aa.set_annotation("occupancy", np.ones(len(aa), dtype=np.float32))

        # Fix NaN b_factors
        nan_b_mask = np.isnan(aa.b_factor)
        if nan_b_mask.any():
            b_factors = aa.b_factor.copy()
            b_factors[nan_b_mask] = 20.0
            aa.set_annotation("b_factor", b_factors)

        reward_inputs = RewardInputs.from_atom_array(aa, ensemble_size=1)
        assert reward_inputs.b_factors.shape[-1] == len(aa)

    def test_nan_coordinates_rejected(self, structure_1vme):
        """NaN coordinates must be caught before constructing reward tensors."""
        atom_array = structure_1vme["asym_unit"].copy()
        # Fix any pre-existing NaN b_factors so the coordinate check passes
        b_factors = np.nan_to_num(atom_array.b_factor, nan=20.0)
        atom_array.set_annotation("b_factor", b_factors)
        coords = atom_array.coord.copy()
        coords[..., 3, :] = np.nan
        atom_array.coord = coords

        with pytest.raises(ValueError, match="NaN coordinates"):
            RewardInputs.from_atom_array(atom_array, ensemble_size=1)


@pytest.fixture
def toy_atom_array() -> AtomArray:
    """Four atoms in two residues, distinct coordinates and B-factors, fractional occupancies."""
    atoms = [
        Atom(
            [1.0, 2.0, 3.0],
            chain_id="A",
            res_id=1,
            res_name="GLY",
            atom_name="N",
            element="N",
            b_factor=11.0,
            occupancy=0.5,
        ),
        Atom(
            [2.0, 3.0, 4.0],
            chain_id="A",
            res_id=1,
            res_name="GLY",
            atom_name="CA",
            element="C",
            b_factor=12.0,
            occupancy=0.5,
        ),
        Atom(
            [3.0, 4.0, 5.0],
            chain_id="A",
            res_id=2,
            res_name="ALA",
            atom_name="N",
            element="N",
            b_factor=13.0,
            occupancy=1.0,
        ),
        Atom(
            [4.0, 5.0, 6.0],
            chain_id="A",
            res_id=2,
            res_name="ALA",
            atom_name="CB",
            element="C",
            b_factor=14.0,
            occupancy=0.25,
        ),
    ]
    return array(atoms)


class TestRewardInputsTopology:
    """``from_atom_array`` keeps the atom array the tensors were built from."""

    def test_topology_is_the_source_array(self, toy_atom_array):
        """The stored topology is the very array passed in, not a copy."""
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=2)

        assert reward_inputs.atom_array is toy_atom_array

    def test_stack_input_keeps_its_first_model(self, toy_atom_array):
        """An ``AtomArrayStack`` shares one topology, represented by its first model."""
        reward_inputs = RewardInputs.from_atom_array(stack([toy_atom_array]), ensemble_size=1)

        assert isinstance(reward_inputs.atom_array, AtomArray)
        assert reward_inputs.atom_array.array_length() == toy_atom_array.array_length()

    def test_tensor_only_construction_has_no_topology(self):
        """Building the dataclass directly, as tests do, leaves ``atom_array`` unset."""
        n_atoms = 3
        reward_inputs = RewardInputs(
            elements=torch.zeros(1, n_atoms, dtype=torch.long),
            b_factors=torch.full((1, n_atoms), 20.0),
            occupancies=torch.ones(1, n_atoms),
            input_coords=torch.zeros(1, n_atoms, 3),
        )

        assert reward_inputs.atom_array is None


class TestRewardInputsToAtomArray:
    """Verify the reference atom array handed to a reward's ``prepare()``."""

    @pytest.mark.parametrize("ensemble_size,num_particles", [(1, 1), (4, 1), (4, 2)])
    def test_round_trip_preserves_coords_b_factors_and_topology(
        self, ensemble_size, num_particles, toy_atom_array
    ):
        """``from_atom_array`` then ``to_atom_array`` returns the template's own values.

        Occupancy is the exception: the result is a topology, so it is 1.0 everywhere,
        whatever the template held.
        """
        reward_inputs = RewardInputs.from_atom_array(
            toy_atom_array, ensemble_size=ensemble_size, num_particles=num_particles
        )

        result = reward_inputs.to_atom_array()

        assert result is not toy_atom_array
        np.testing.assert_allclose(result.coord, toy_atom_array.coord)
        np.testing.assert_allclose(result.b_factor, toy_atom_array.b_factor)
        np.testing.assert_allclose(result.occupancy, np.ones(toy_atom_array.array_length()))
        for category in ["chain_id", "res_id", "res_name", "atom_name", "element"]:
            np.testing.assert_array_equal(
                result.get_annotation(category), toy_atom_array.get_annotation(category)
            )
        # The template is read, never written: its fractional occupancies survive.
        np.testing.assert_allclose(toy_atom_array.occupancy, [0.5, 0.5, 1.0, 0.25])

    def test_reconciled_values_override_the_template(self, toy_atom_array):
        """Coordinates and B-factors come from the tensors, as ``to_reward_inputs`` sets them.

        This is the handoff that matters for ``prepare()``: a model template's placeholder
        B-factors and coordinates must not reach a forward model built from this array.
        """
        n_atoms = toy_atom_array.array_length()
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=2)
        new_b_factors = torch.arange(n_atoms, dtype=torch.float32) + 100.0
        new_coords = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3) - 50.0
        reconciled = replace(
            reward_inputs,
            b_factors=new_b_factors.expand(2, -1),
            input_coords=new_coords.expand(2, -1, -1),
        )

        result = reconciled.to_atom_array()

        np.testing.assert_allclose(result.b_factor, new_b_factors.numpy())
        np.testing.assert_allclose(result.coord, new_coords.numpy())
        assert reconciled.atom_array is toy_atom_array
        np.testing.assert_allclose(toy_atom_array.b_factor, [11.0, 12.0, 13.0, 14.0])

    def test_explicit_template_overrides_the_stored_topology(self, toy_atom_array):
        """A caller may supply the topology array instead of relying on the stored one."""
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=1)
        template = toy_atom_array.copy()
        template.set_annotation("atom_name", np.array(["X1", "X2", "X3", "X4"]))

        result = reward_inputs.to_atom_array(template)

        np.testing.assert_array_equal(result.atom_name, ["X1", "X2", "X3", "X4"])
        np.testing.assert_allclose(result.coord, toy_atom_array.coord)

    def test_template_atom_count_mismatch_raises(self, toy_atom_array):
        """A template that does not describe these atoms is rejected."""
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=1)

        with pytest.raises(ValueError, match="atoms"):
            reward_inputs.to_atom_array(toy_atom_array[:2])

    def test_without_any_topology_raises(self):
        """Tensor-only inputs cannot produce an atom array unless a template is given."""
        n_atoms = 4
        reward_inputs = RewardInputs(
            elements=torch.zeros(1, n_atoms, dtype=torch.long),
            b_factors=torch.full((1, n_atoms), 20.0),
            occupancies=torch.ones(1, n_atoms),
            input_coords=torch.zeros(1, n_atoms, 3),
        )

        with pytest.raises(ValueError, match="no atom array"):
            reward_inputs.to_atom_array()
