"""Tests for LossInputs.

``from_atom_array`` must reject atom arrays with NaN B-factors, NaN coordinates, and
invalid occupancies, and keep the topology it was built from. ``to_atom_array`` must
rebuild the reference array a loss's ``prepare()`` needs without touching that
topology.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from biotite.structure import array, Atom, AtomArray, stack
from sampleworks.core.losses.protocol import LossInputs
from sampleworks.utils.elements import elements_to_scattering_indices


class TestLossInputsFromAtomArray:
    """Validate that LossInputs.from_atom_array rejects invalid atom arrays."""

    def test_nan_b_factors_from_missing_atoms_rejected(self, atom_array_1vme_with_missing_atoms):
        """
        LossInputs rejects the raw atom array from add_missing_atoms.
        """
        with pytest.raises(ValueError, match="NaN B-factors"):
            LossInputs.from_atom_array(atom_array_1vme_with_missing_atoms, ensemble_size=1)

    def test_cleaned_missing_atoms_accepted(self, atom_array_1vme_with_missing_atoms):
        """Once a wrapper cleans the array, the tensors carry exactly the cleaned values.

        Cleaning follows the contract in the ``LossInputs`` docstring: finite
        coordinates, occupancy 1.0, and the documented default B-factor of 20.0 for
        atoms that had none.
        """
        aa = atom_array_1vme_with_missing_atoms.copy()
        nan_coord_mask = np.any(np.isnan(aa.coord), axis=-1)  # (n_atoms,)
        nan_b_mask = np.isnan(aa.b_factor)  # (n_atoms,)
        assert nan_coord_mask.any() and nan_b_mask.any(), "fixture must have missing atoms"

        cleaned_coords = aa.coord.copy()  # (n_atoms, 3)
        cleaned_coords[nan_coord_mask] = cleaned_coords[~nan_coord_mask].mean(axis=0)
        aa.coord = cleaned_coords
        aa.set_annotation("occupancy", np.ones(len(aa), dtype=np.float32))
        aa.set_annotation("b_factor", np.where(nan_b_mask, 20.0, aa.b_factor).astype(np.float32))

        loss_inputs = LossInputs.from_atom_array(aa, ensemble_size=1)

        np.testing.assert_allclose(loss_inputs.input_coords[0].numpy(), cleaned_coords)
        b_factors = loss_inputs.b_factors[0].numpy()  # (n_atoms,)
        np.testing.assert_allclose(b_factors[nan_b_mask], 20.0)
        np.testing.assert_allclose(
            b_factors[~nan_b_mask], atom_array_1vme_with_missing_atoms.b_factor[~nan_b_mask]
        )

    @pytest.mark.parametrize("bad_occupancy", [1.5, -0.1])
    def test_occupancy_outside_unit_interval_rejected(self, toy_atom_array, bad_occupancy):
        """Occupancy is a fraction; values above 1 or below 0 are rejected."""
        atom_array = toy_atom_array.copy()
        occupancy = atom_array.occupancy.copy()
        occupancy[2] = bad_occupancy
        atom_array.set_annotation("occupancy", occupancy)

        with pytest.raises(ValueError, match="invalid occupancy"):
            LossInputs.from_atom_array(atom_array, ensemble_size=1)

    def test_nan_coordinates_rejected(self, structure_1vme):
        """NaN coordinates must be caught before constructing loss tensors."""
        atom_array = structure_1vme["asym_unit"].copy()
        # Fix any pre-existing NaN b_factors so the coordinate check passes
        b_factors = np.nan_to_num(atom_array.b_factor, nan=20.0)
        atom_array.set_annotation("b_factor", b_factors)
        coords = atom_array.coord.copy()
        coords[..., 3, :] = np.nan
        atom_array.coord = coords

        with pytest.raises(ValueError, match="NaN coordinates"):
            LossInputs.from_atom_array(atom_array, ensemble_size=1)


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


class TestLossInputsTensorValues:
    """The tensors hold the array's per-atom values, tiled over ensemble and particles."""

    @pytest.mark.parametrize("ensemble_size,num_particles", [(1, 1), (4, 1), (4, 2)])
    def test_tensors_tile_per_atom_values_over_the_batch(
        self, toy_atom_array, ensemble_size, num_particles
    ):
        """Each ensemble member carries weight ``1/ensemble_size``, whatever the file says.

        The ensemble is the model's population, so occupancy is the uniform mixture
        weight over its members (summing to 1 per atom within a particle), not the
        deposited occupancy (0.5, 0.5, 1.0, 0.25 here). Elements, B-factors and
        coordinates are the array's own, repeated for every member and particle.
        """
        loss_inputs = LossInputs.from_atom_array(
            toy_atom_array, ensemble_size=ensemble_size, num_particles=num_particles
        )

        n_atoms = toy_atom_array.array_length()
        batch_shape = (num_particles, ensemble_size) if num_particles > 1 else (ensemble_size,)
        nitrogen, carbon = elements_to_scattering_indices(np.array(["N", "C"]))
        expected_elements = torch.tensor([nitrogen, carbon, nitrogen, carbon]).expand(
            *batch_shape, n_atoms
        )  # (*batch, n_atoms)
        expected_b_factors = torch.tensor([11.0, 12.0, 13.0, 14.0]).expand(*batch_shape, n_atoms)

        assert nitrogen != carbon
        torch.testing.assert_close(loss_inputs.elements, expected_elements, check_dtype=False)
        torch.testing.assert_close(loss_inputs.b_factors, expected_b_factors)
        torch.testing.assert_close(
            loss_inputs.occupancies, torch.full((*batch_shape, n_atoms), 1.0 / ensemble_size)
        )
        torch.testing.assert_close(
            loss_inputs.occupancies.sum(dim=-2), torch.ones(*batch_shape[:-1], n_atoms)
        )
        # Coordinates are flat over (particles * ensemble); FK steering adds noise per row.
        expected_coords = torch.as_tensor(toy_atom_array.coord).expand(
            num_particles * ensemble_size, n_atoms, 3
        )  # (particles * ensemble, n_atoms, 3)
        torch.testing.assert_close(loss_inputs.input_coords, expected_coords)


class TestLossInputsTopology:
    """``from_atom_array`` keeps the atom array the tensors were built from."""

    def test_topology_is_the_source_array(self, toy_atom_array):
        """The stored topology is the very array passed in, not a copy."""
        loss_inputs = LossInputs.from_atom_array(toy_atom_array, ensemble_size=2)

        assert loss_inputs.atom_array is toy_atom_array

    def test_stack_input_keeps_its_first_model(self, toy_atom_array):
        """An ``AtomArrayStack`` shares one topology, represented by its first model."""
        loss_inputs = LossInputs.from_atom_array(stack([toy_atom_array]), ensemble_size=1)

        assert isinstance(loss_inputs.atom_array, AtomArray)
        assert loss_inputs.atom_array.array_length() == toy_atom_array.array_length()

    def test_tensor_only_construction_has_no_topology(self):
        """Building the dataclass directly, as tests do, leaves ``atom_array`` unset."""
        n_atoms = 3
        loss_inputs = LossInputs(
            elements=torch.zeros(1, n_atoms, dtype=torch.long),
            b_factors=torch.full((1, n_atoms), 20.0),
            occupancies=torch.ones(1, n_atoms),
            input_coords=torch.zeros(1, n_atoms, 3),
        )

        assert loss_inputs.atom_array is None


class TestLossInputsToAtomArray:
    """Verify the reference atom array handed to a loss's ``prepare()``."""

    @pytest.mark.parametrize("ensemble_size,num_particles", [(1, 1), (4, 1), (4, 2)])
    def test_round_trip_preserves_coords_b_factors_and_topology(
        self, ensemble_size, num_particles, toy_atom_array
    ):
        """``from_atom_array`` then ``to_atom_array`` returns the template's own values.

        Occupancy is the exception: the result is a topology, so it is 1.0 everywhere,
        whatever the template held.
        """
        loss_inputs = LossInputs.from_atom_array(
            toy_atom_array, ensemble_size=ensemble_size, num_particles=num_particles
        )

        result = loss_inputs.to_atom_array()

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
        """Coordinates and B-factors come from the tensors, as ``to_loss_inputs`` sets them.

        This is the handoff that matters for ``prepare()``: a model template's placeholder
        B-factors and coordinates must not reach a forward model built from this array.
        """
        n_atoms = toy_atom_array.array_length()
        loss_inputs = LossInputs.from_atom_array(toy_atom_array, ensemble_size=2)
        new_b_factors = torch.arange(n_atoms, dtype=torch.float32) + 100.0
        new_coords = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3) - 50.0
        reconciled = replace(
            loss_inputs,
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
        loss_inputs = LossInputs.from_atom_array(toy_atom_array, ensemble_size=1)
        template = toy_atom_array.copy()
        template.set_annotation("atom_name", np.array(["X1", "X2", "X3", "X4"]))

        result = loss_inputs.to_atom_array(template)

        np.testing.assert_array_equal(result.atom_name, ["X1", "X2", "X3", "X4"])
        np.testing.assert_allclose(result.coord, toy_atom_array.coord)

    def test_template_atom_count_mismatch_raises(self, toy_atom_array):
        """A template that does not describe these atoms is rejected."""
        loss_inputs = LossInputs.from_atom_array(toy_atom_array, ensemble_size=1)

        with pytest.raises(ValueError, match="atoms"):
            loss_inputs.to_atom_array(toy_atom_array[:2])

    def test_without_any_topology_raises(self):
        """Tensor-only inputs cannot produce an atom array unless a template is given."""
        n_atoms = 4
        loss_inputs = LossInputs(
            elements=torch.zeros(1, n_atoms, dtype=torch.long),
            b_factors=torch.full((1, n_atoms), 20.0),
            occupancies=torch.ones(1, n_atoms),
            input_coords=torch.zeros(1, n_atoms, 3),
        )

        with pytest.raises(ValueError, match="no atom array"):
            loss_inputs.to_atom_array()
