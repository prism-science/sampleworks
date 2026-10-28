"""Tests for atom_array_utils module."""

from typing import Any, cast

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.filter import filter_polymer
from biotite.structure.info.groups import amino_acid_names
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    keep_amino_acids,
    keep_polymer,
    make_normalized_atom_id,
    remove_atoms_with_any_nan_coords,
    remove_hydrogens,
    select_altloc,
)

from tests.conftest import STRUCTURES


STANDARD_AA_NAMES = frozenset(amino_acid_names())


class TestSelectAltlocBasic:
    """Basic functionality tests for select_altloc."""

    def test_select_specific_altloc(self, basic_atom_array_altloc):
        """Test selecting atoms with a specific altloc_id."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)

        assert len(result) == 2
        assert all(result.altloc_id == "A")
        assert list(cast(np.ndarray, result.res_name)) == ["ALA", "VAL"]

    def test_select_different_altloc(self, basic_atom_array_altloc):
        """Test selecting different altloc_id values."""
        result_b = select_altloc(basic_atom_array_altloc, "B", return_full_array=False)
        result_c = select_altloc(basic_atom_array_altloc, "C", return_full_array=False)

        assert len(result_b) == 2
        assert all(result_b.altloc_id == "B")
        assert len(result_c) == 1
        assert all(result_c.altloc_id == "C")

    def test_select_nonexistent_altloc(self, basic_atom_array_altloc):
        """Test selecting altloc that doesn't exist returns empty array."""
        result = select_altloc(basic_atom_array_altloc, "Z", return_full_array=False)

        assert len(result) == 0

    def test_preserves_coordinates(self, basic_atom_array_altloc):
        """Test that selected atoms preserve their original coordinates."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)
        result_coord = cast(np.ndarray, result.coord)
        basic_coord = cast(np.ndarray, basic_atom_array_altloc.coord)

        np.testing.assert_array_equal(result_coord[0], basic_coord[0])
        np.testing.assert_array_equal(result_coord[1], basic_coord[2])

    def test_preserves_annotations(self, basic_atom_array_altloc):
        """Test that other annotations are preserved."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)

        assert list(cast(np.ndarray, result.chain_id)) == ["A", "A"]
        assert list(cast(np.ndarray, result.res_id)) == [1, 2]
        assert list(cast(np.ndarray, result.atom_name)) == ["CA", "CA"]

    def test_6b8x_select_altloc_a(self, structure_6b8x_with_altlocs):
        """PDB 6b8x has 695 atoms in altloc A."""
        arr = structure_6b8x_with_altlocs
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        result = select_altloc(arr, "A", return_full_array=False)

        assert len(result) == 695
        assert all(result.altloc_id == "A")

    def test_6b8x_select_rare_altloc_c(self, structure_6b8x_with_altlocs):
        """PDB 6b8x altloc C is sparse: 46 atoms across residues 150-153 and 224."""
        arr = structure_6b8x_with_altlocs
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        result = select_altloc(arr, "C", return_full_array=False)

        assert len(result) == 46
        assert set(cast(np.ndarray, result.res_id)) == {150, 151, 152, 153, 224}

    def test_6b8x_full_array_includes_blank_altlocs(self, structure_6b8x_with_altlocs):
        """return_full_array=True includes both target altloc and non-altloc atoms."""
        arr = structure_6b8x_with_altlocs
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        result = select_altloc(arr, "A", return_full_array=True)

        # 695 altloc-A atoms + 1906 blank-altloc atoms = 2601
        assert len(result) == 2601
        unique_altlocs = set(cast(np.ndarray, result.altloc_id))
        assert "A" in unique_altlocs
        assert "B" not in unique_altlocs
        assert "C" not in unique_altlocs

    def test_6b8x_occupancies_preserved(self, structure_6b8x_with_altlocs):
        """Occupancy values are preserved after altloc selection on real data."""
        arr = structure_6b8x_with_altlocs
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        result = select_altloc(arr, "A", return_full_array=False)
        occupancies = cast(np.ndarray, result.occupancy)

        assert occupancies.min() > 0.0
        assert occupancies.max() <= 1.0
        # 6b8x altloc A occupancies span 0.18–1.00
        assert occupancies.min() == pytest.approx(0.18, abs=0.01)
        assert occupancies.max() == pytest.approx(1.00, abs=0.01)


class TestSelectAltlocWithStack:
    """Tests for AtomArrayStack inputs."""

    def test_select_from_stack(self, atom_array_stack_altloc):
        """Test selecting altloc from AtomArrayStack."""
        result = select_altloc(atom_array_stack_altloc, "A", return_full_array=False)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 3
        assert result.array_length() == 2
        assert all(result[0].altloc_id == "A")
        assert all(result[1].altloc_id == "A")
        assert all(result[2].altloc_id == "A")

    def test_select_from_stack_with_full_array(self, atom_array_stack_altloc):
        """Test selecting from stack with return_full_array=True."""
        result = select_altloc(atom_array_stack_altloc, "A", return_full_array=True)
        result_stack = cast(AtomArrayStack, result)

        # Should include: 2 "A" atoms
        assert result_stack.stack_depth() == 3
        assert result_stack.array_length() == 2

    def test_stack_preserves_all_models(self, atom_array_stack_altloc):
        """Test that all models in stack are preserved after filtering."""
        result = select_altloc(atom_array_stack_altloc, "B", return_full_array=False)
        result_stack = cast(AtomArrayStack, result)

        assert result_stack.stack_depth() == atom_array_stack_altloc.stack_depth()
        # Each model should have 2 "B" atoms
        for i in range(result_stack.stack_depth()):
            model = cast(AtomArray, result_stack[i])
            assert len(model) == 2
            assert all(model.altloc_id == "B")

    def test_6b8x_stack_select(self, structure_6b8x_with_altlocs):
        """select_altloc on a real AtomArrayStack preserves stack type and depth."""
        arr = structure_6b8x_with_altlocs
        if not isinstance(arr, AtomArrayStack):
            pytest.skip("fixture is not a stack in this environment")

        result = select_altloc(arr, "A", return_full_array=False)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == arr.stack_depth()
        assert result.array_length() == 695


class TestSelectAltlocErrors:
    """Tests for error handling."""

    def test_missing_altloc_id_raises_error(self, atom_array_missing_altloc_id):
        """Test that missing altloc_id annotation raises AttributeError."""
        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array_missing_altloc_id, "A")

    def test_missing_occupancy_raises_error(self, atom_array_missing_occupancy):
        """Test that missing occupancy annotation raises AttributeError."""
        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array_missing_occupancy, "A")

    def test_missing_both_attributes_raises_error(self):
        """Test that missing both altloc_id and occupancy raises AttributeError."""
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3)

        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array, "A")

    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises TypeError."""
        invalid_input = "not an atom array"

        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(cast(Any, invalid_input), "A")

    def test_none_input_raises_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(cast(Any, None), "A")


class TestSelectAltlocEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Test with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.altloc_id = np.array([], dtype=str)
        atom_array.occupancy = np.array([], dtype=float)

        result = select_altloc(atom_array, "A", return_full_array=False)

        assert len(result) == 0

    def test_all_same_altloc(self):
        """Test when all atoms have the same altloc_id."""
        atom_array = AtomArray(4)
        atom_array.set_annotation("altloc_id", np.array(["A", "A", "A", "A"]))
        atom_array.set_annotation("occupancy", np.array([0.5, 0.6, 0.7, 0.8]))

        result = select_altloc(atom_array, "A", return_full_array=False)

        assert len(result) == 4
        assert all(result.altloc_id == "A")


class TestFilterToCommonAtoms:
    """Tests for filter_to_common_atoms function."""

    def test_filters_to_common_atoms(self, atom_array_partial_overlap):
        """Test that only common atoms are returned."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Should have 3 common atoms (residues 3, 4, 5)
        assert filtered1.array_length() == 3
        assert filtered2.array_length() == 3
        assert list(cast(np.ndarray, filtered1.res_id)) == [3, 4, 5]
        assert list(cast(np.ndarray, filtered2.res_id)) == [3, 4, 5]

    def test_atoms_in_matching_order(self, atom_array_partial_overlap):
        """Test that returned atoms are in matching order."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Check that atom identifiers match
        assert np.array_equal(
            cast(np.ndarray, filtered1.chain_id), cast(np.ndarray, filtered2.chain_id)
        )
        assert np.array_equal(
            cast(np.ndarray, filtered1.res_id), cast(np.ndarray, filtered2.res_id)
        )
        assert np.array_equal(
            cast(np.ndarray, filtered1.atom_name), cast(np.ndarray, filtered2.atom_name)
        )

    def test_with_identical_arrays(self, basic_atom_array_altloc):
        """Test with two identical arrays."""
        filtered1, filtered2 = filter_to_common_atoms(
            basic_atom_array_altloc, basic_atom_array_altloc
        )

        # Should return all atoms
        assert filtered1.array_length() == len(basic_atom_array_altloc)
        assert filtered2.array_length() == len(basic_atom_array_altloc)

    def test_with_no_overlap(self):
        """Test with arrays that have no common atoms."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["A"] * 3))
        array2.set_annotation("res_id", np.array([4, 5, 6]))
        array2.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No common atoms found across all 2 structures"):
            filter_to_common_atoms(array1, array2)

    def test_with_stacks(self, atom_array_stacks_partial_overlap):
        """Test with AtomArrayStacks."""
        stack1, stack2 = atom_array_stacks_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(stack1, stack2)

        # Should have 3 common atoms (residues 2, 3, 4)
        assert isinstance(filtered1, AtomArrayStack)
        assert isinstance(filtered2, AtomArrayStack)
        assert filtered1.stack_depth() == 2
        assert filtered2.stack_depth() == 2
        assert filtered1.array_length() == 3
        assert filtered2.array_length() == 3

    def test_mixed_array_and_stack(self, basic_atom_array_altloc, atom_array_stack_altloc):
        """Test with one AtomArray and one AtomArrayStack."""
        # Both have same res_ids 1-5
        filtered1, filtered2 = filter_to_common_atoms(
            basic_atom_array_altloc, atom_array_stack_altloc
        )

        assert isinstance(filtered1, AtomArrayStack)
        assert isinstance(filtered2, AtomArrayStack)
        assert filtered1.array_length() == 5
        assert filtered2.array_length() == 5

    def test_different_chains_no_overlap(self):
        """Test arrays with different chain IDs have no overlap."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["B"] * 3))
        array2.set_annotation("res_id", np.array([1, 2, 3]))
        array2.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No common atoms found"):
            filter_to_common_atoms(array1, array2)

    def test_different_atom_names_no_overlap(self):
        """Test arrays with different atom names have no overlap."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["A"] * 3))
        array2.set_annotation("res_id", np.array([1, 2, 3]))
        array2.set_annotation("atom_name", np.array(["CB", "CB", "CB"]))

        with pytest.raises(RuntimeError, match="No common atoms found"):
            filter_to_common_atoms(array1, array2)

    def test_invalid_type_first_arg(self):
        """Test that invalid first argument raises TypeError."""
        array = AtomArray(3)
        array.coord = np.random.rand(3, 3)

        with pytest.raises(TypeError, match="must be AtomArray or AtomArrayStack"):
            filter_to_common_atoms(cast(Any, "not an array"), array)

    def test_invalid_type_second_arg(self):
        """Test that invalid second argument raises TypeError."""
        array = AtomArray(3)
        array.coord = np.random.rand(3, 3)

        with pytest.raises(TypeError, match="must be AtomArray or AtomArrayStack"):
            filter_to_common_atoms(array, cast(Any, None))

    def test_preserves_coordinates(self, atom_array_partial_overlap):
        """Test that coordinates are preserved for common atoms."""
        # TODO filter_to_common_atoms now returns a tuple of AtomArrayStack
        array1, array2 = atom_array_partial_overlap
        original_coords1 = array1.coord.copy()
        original_coords2 = array2.coord.copy()

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        assert filtered1 is not None and filtered1.coord is not None
        assert filtered2 is not None and filtered2.coord is not None

        # Filtered arrays should have coordinates from residues 3, 4, 5
        # which are indices 2, 3, 4 in array1 and 0, 1, 2 in array2
        np.testing.assert_array_equal(filtered1.coord[0, 0], original_coords1[2])
        np.testing.assert_array_equal(filtered2.coord[0, 0], original_coords2[0])


def _get_atom_array(structure: dict) -> AtomArray:
    arr = structure["asym_unit"]
    if isinstance(arr, AtomArrayStack):
        return cast(AtomArray, arr[0])
    return cast(AtomArray, arr)


def _get_atom_array_stack(structure: dict) -> AtomArrayStack:
    arr = structure["asym_unit"]
    if isinstance(arr, AtomArray):
        return stack([arr, arr])
    return cast(AtomArrayStack, arr)


class TestRemoveHydrogens:
    """Tests for remove_hydrogens function."""

    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_no_hydrogen_or_deuterium_in_output(self, structure_fixture, request):
        """Output must contain zero H or D and preserve all heavy atoms."""
        structure = request.getfixturevalue(structure_fixture)
        arr = _get_atom_array(structure)
        result = remove_hydrogens(arr)
        elements = cast(np.ndarray, result.element)
        assert "H" not in elements
        assert "D" not in elements
        input_elements = cast(np.ndarray, arr.element)
        expected_count = int(np.sum((input_elements != "H") & (input_elements != "D")))
        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_atom_array_and_stack_give_same_atom_count(self, structure_fixture, request):
        """AtomArray and AtomArrayStack paths must select the same atoms."""
        structure = request.getfixturevalue(structure_fixture)
        arr_result = remove_hydrogens(_get_atom_array(structure))
        stack_result = remove_hydrogens(_get_atom_array_stack(structure))

        assert isinstance(stack_result, AtomArrayStack)
        assert len(arr_result) == stack_result.array_length()
        assert "H" not in cast(np.ndarray, stack_result.element)

    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_is_idempotent(self, structure_fixture, request):
        """Property: applying remove_hydrogens twice == applying once."""
        structure = request.getfixturevalue(structure_fixture)
        once = remove_hydrogens(_get_atom_array(structure))
        twice = remove_hydrogens(once)
        assert len(once) == len(twice)

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            remove_hydrogens(cast(Any, "bad input"))


class TestKeepPolymer:
    """Tests for keep_polymer function."""

    def test_output_contains_only_polymer_atoms(self, test_structure):
        """Every atom in the result must pass filter_polymer."""
        arr = _get_atom_array(test_structure)
        result = keep_polymer(arr)
        assert len(result) > 0
        polymer_mask = filter_polymer(result)
        assert np.all(polymer_mask)

    def test_atom_array_and_stack_give_same_atom_count(self, test_structure):
        """Both AtomArray and AtomArrayStack should work."""
        arr_result = keep_polymer(_get_atom_array(test_structure))
        stack_result = keep_polymer(_get_atom_array_stack(test_structure))

        assert isinstance(stack_result, AtomArrayStack)
        assert len(arr_result) == stack_result.array_length()

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            keep_polymer(cast(Any, 42))


class TestKeepAminoAcids:
    """Tests for keep_amino_acids function."""

    def test_output_contains_only_standard_amino_acid_residues(self, test_structure):
        """Purity (only AA residues) and completeness (all AA atoms preserved)."""
        arr = _get_atom_array(test_structure)
        result = keep_amino_acids(arr)
        result_names = set(cast(np.ndarray, result.res_name))
        assert len(result) > 0
        assert result_names <= STANDARD_AA_NAMES
        input_names = cast(np.ndarray, arr.res_name)
        expected_count = int(np.sum(np.isin(input_names, list(STANDARD_AA_NAMES))))
        assert len(result) == expected_count

    def test_atom_array_and_stack_give_same_atom_count(self, test_structure):
        """AtomArray and AtomArrayStack paths must select the same atoms."""
        arr_result = keep_amino_acids(_get_atom_array(test_structure))
        stack_result = keep_amino_acids(_get_atom_array_stack(test_structure))

        assert isinstance(stack_result, AtomArrayStack)
        assert len(arr_result) == stack_result.array_length()

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            keep_amino_acids(cast(Any, None))


class TestFilterFunctionsIntegration:
    """Property and composition tests across multiple filter functions."""

    def test_remove_hydrogens_and_keep_amino_acids_commute(self, test_structure):
        """These filters act on independent atom properties so order shouldn't matter."""
        arr = _get_atom_array(test_structure)
        path_a = keep_amino_acids(remove_hydrogens(arr))
        path_b = remove_hydrogens(keep_amino_acids(arr))
        assert len(path_a) == len(path_b)
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.res_id), cast(np.ndarray, path_b.res_id)
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.atom_name), cast(np.ndarray, path_b.atom_name)
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.element), cast(np.ndarray, path_b.element)
        )

    def test_polymer_and_remove_hydrogens_commute(self, test_structure):
        """Filter order must not change the result:
        polymer(dehydrogenate(array) == dehydrogenate(polymer(array))."""
        arr = _get_atom_array(test_structure)
        polymer_first = remove_hydrogens(keep_polymer(arr))
        hydrogen_first = keep_polymer(remove_hydrogens(arr))

        assert len(polymer_first) > 0
        assert len(polymer_first) == len(hydrogen_first)

        # Same atoms selected regardless of order
        np.testing.assert_array_equal(
            cast(np.ndarray, polymer_first.chain_id),
            cast(np.ndarray, hydrogen_first.chain_id),
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, polymer_first.res_id),
            cast(np.ndarray, hydrogen_first.res_id),
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, polymer_first.atom_name),
            cast(np.ndarray, hydrogen_first.atom_name),
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, polymer_first.element),
            cast(np.ndarray, hydrogen_first.element),
        )

        # Coordinates and atom order are identical
        np.testing.assert_array_equal(polymer_first.coord, hydrogen_first.coord)

        # Output satisfies both filters' postconditions
        elements = cast(np.ndarray, polymer_first.element)
        assert "H" not in elements
        assert "D" not in elements
        assert np.all(filter_polymer(polymer_first))

    def test_6b8x_altloc_a_vs_b_common_atoms(self, structure_6b8x_with_altlocs):
        """Full-array altloc A and B share atoms at positions with blank altlocs."""
        arr = structure_6b8x_with_altlocs
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        full_a = select_altloc(arr, "A", return_full_array=True)
        full_b = select_altloc(arr, "B", return_full_array=True)

        f_a, f_b = filter_to_common_atoms(full_a, full_b)

        # Both full arrays share the 1906 blank-altloc atoms plus overlapping
        # altloc positions → 2447 common atoms (empirically verified)
        assert f_a.array_length() == 2447
        assert f_b.array_length() == 2447


class TestNormalizedAtomId:
    """Tests for make_normalized_atom_id."""

    def test_different_numbering_same_result(
        self, backbone_two_residues, backbone_two_residues_offset
    ):
        """Two arrays with identical atoms but offset res_id produce the same IDs."""
        np.testing.assert_array_equal(
            make_normalized_atom_id(backbone_two_residues),
            make_normalized_atom_id(backbone_two_residues_offset),
        )

    def test_different_chains_differ(self, two_chain_array):
        """Atoms on different chains get distinct normalized IDs."""
        ids = make_normalized_atom_id(two_chain_array)
        assert ids[0] != ids[1]

    def test_reordered_chains_same_ids(self, reordered_chains):
        """Chain ordering [B,A] vs [A,B] gives each chain's atom the same ID in both arrays."""
        model, struct = reordered_chains
        model_ids = make_normalized_atom_id(model)
        struct_ids = make_normalized_atom_id(struct)
        model_chains = cast(np.ndarray, model.chain_id)
        struct_chains = cast(np.ndarray, struct.chain_id)
        for chain in ("A", "B"):
            np.testing.assert_array_equal(
                model_ids[model_chains == chain], struct_ids[struct_chains == chain]
            )
        assert model_ids[0] != model_ids[1]

    def test_1vme_symmetric_chains_produce_same_normalized_ids(self, structure_1vme):
        """1vme chains A and B are symmetric homodimers with identical topology."""
        arr = structure_1vme["asym_unit"]
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        chain_a = arr[arr.chain_id == "A"]
        chain_b = arr[arr.chain_id == "B"]
        chain_b.chain_id[:] = "A"  # ids carry the chain name

        ids_a = sorted(make_normalized_atom_id(chain_a))
        ids_b = sorted(make_normalized_atom_id(chain_b))

        np.testing.assert_array_equal(ids_a, ids_b)

    def test_6b8x_unique_ids_per_atom(self, structure_6b8x):
        """Every atom in 6b8x gets a unique normalized ID (single-chain, no duplication)."""
        arr = structure_6b8x["asym_unit"]
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        ids = make_normalized_atom_id(arr)

        assert len(ids) == len(set(ids))


class TestFilterToCommonAtomsNormalized:
    """Tests for filter_to_common_atoms with normalize_ids=True."""

    def test_different_numbering(self, model_struct_numbering_offset):
        """Same atoms with different res_id schemes are matched correctly."""
        model, struct = model_struct_numbering_offset
        (fm, fs) = filter_to_common_atoms(model, struct, normalize_ids=True)
        assert fm.array_length() == 3
        assert fs.array_length() == 3
        fm_arr, fs_arr = cast(AtomArray, fm[0]), cast(AtomArray, fs[0])
        fm_names = cast(np.ndarray, fm_arr.atom_name)
        fs_names = cast(np.ndarray, fs_arr.atom_name)
        for i in range(len(fm_arr)):
            assert fm_names[i] == fs_names[i]

    def test_without_normalize_fails_on_different_numbering(self, single_atom_numbering_offset):
        """Without normalize_ids, different res_id numbering finds no overlap."""
        model, struct = single_atom_numbering_offset
        with pytest.raises(RuntimeError, match="No common atoms"):
            filter_to_common_atoms(model, struct, normalize_ids=False)

    def test_multi_chain(self, multi_chain_numbering_offset):
        """Normalized matching works across multiple chains."""
        model, struct = multi_chain_numbering_offset
        (fm, fs) = filter_to_common_atoms(model, struct, normalize_ids=True)
        assert fm.array_length() == 3

    def test_reordered_chains(self, reordered_chains):
        """Chain ordering [B,A] vs [A,B] pairs chain A with A and B with B, not by position."""
        model, struct = reordered_chains
        (fm, fs), (m_idx, s_idx) = filter_to_common_atoms(
            model, struct, normalize_ids=True, return_indices=True
        )
        assert fm.array_length() == 2
        fm_chains = fm.chain_id
        fs_chains = fs.chain_id
        np.testing.assert_array_equal(fm_chains, fs_chains)
        assert set(fm_chains) == {"A", "B"}
        model_chains = model.chain_id
        struct_chains = struct.chain_id
        np.testing.assert_array_equal(model_chains[m_idx], struct_chains[s_idx])

    def test_1vme_normalized_atom_names_align(self, structure_1vme):
        """Normalized matching of 1vme chain A vs B produces atom-level alignment."""
        arr = structure_1vme["asym_unit"]
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        chain_a = arr[arr.chain_id == "A"]
        chain_b = arr[arr.chain_id == "B"]
        chain_b.chain_id[:] = "A"  # chains pair by name

        (fa, fb) = filter_to_common_atoms(chain_a, chain_b, normalize_ids=True)
        fa_arr = cast(AtomArray, fa[0])
        fb_arr = cast(AtomArray, fb[0])

        np.testing.assert_array_equal(
            cast(np.ndarray, fa_arr.atom_name), cast(np.ndarray, fb_arr.atom_name)
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, fa_arr.res_name), cast(np.ndarray, fb_arr.res_name)
        )


class TestReturnIndices:
    """Tests for filter_to_common_atoms with return_indices=True."""

    def test_indices_point_to_correct_atoms(self, model_struct_partial_atom_names):
        """Returned indices map back to the correct atoms in the originals."""
        model, struct = model_struct_partial_atom_names
        (_, _), (m_idx, s_idx) = filter_to_common_atoms(
            model, struct, normalize_ids=True, return_indices=True
        )
        assert len(m_idx) == 2
        for mi, si in zip(m_idx, s_idx):
            assert model.atom_name[mi] == struct.atom_name[si]

    def test_backward_compat_without_return_indices(self, backbone_two_residues):
        """Default call (return_indices=False) returns only filtered arrays."""
        result = filter_to_common_atoms(backbone_two_residues, backbone_two_residues)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert hasattr(result[0], "coord")

    def test_6b8x_indices_roundtrip(self, structure_6b8x):
        """Returned indices reconstruct the filtered atoms from the original 6b8x array."""
        arr = structure_6b8x["asym_unit"]
        if isinstance(arr, AtomArrayStack):
            arr = cast(AtomArray, arr[0])

        # Overlapping slices: sub1=[0:200], sub2=[100:end] → overlap at [100:200]
        sub1 = arr[:200]
        sub2 = arr[100:]

        (f1, f2), (idx1, idx2) = filter_to_common_atoms(sub1, sub2, return_indices=True)

        assert len(idx1) == len(idx2)
        for i1, i2 in zip(idx1, idx2):
            assert sub1.chain_id[i1] == sub2.chain_id[i2]
            assert sub1.res_id[i1] == sub2.res_id[i2]
            assert sub1.atom_name[i1] == sub2.atom_name[i2]


class TestRemoveAtomsWithAnyNanCoords:
    """Tests for remove_atoms_with_any_nan_coords function."""

    def test_atom_array_all_finite(self):
        """Test AtomArray with all finite coordinates - should return unchanged."""
        atom_array = AtomArray(5)
        atom_array.coord = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ]
        )
        atom_array.set_annotation("chain_id", np.array(["A"] * 5))
        atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
        atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))

        result = remove_atoms_with_any_nan_coords(atom_array)

        assert len(result) == 5
        np.testing.assert_array_equal(result.coord, atom_array.coord)

    def test_atom_array_with_nan_coords(self):
        """Test AtomArray with some NaN coordinates - should remove those atoms."""
        atom_array = AtomArray(5)
        atom_array.coord = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, np.nan, 6.0],  # Has NaN in y coordinate
                [7.0, 8.0, 9.0],
                [np.nan, 11.0, 12.0],  # Has NaN in x coordinate
                [13.0, 14.0, 15.0],
            ]
        )
        atom_array.set_annotation("chain_id", np.array(["A"] * 5))
        atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
        atom_array.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD", "CE"]))

        result = remove_atoms_with_any_nan_coords(atom_array)

        assert len(result) == 3
        assert list(cast(np.ndarray, result.res_id)) == [1, 3, 5]
        assert list(cast(np.ndarray, result.atom_name)) == ["CA", "CG", "CE"]
        assert np.isfinite(result.coord).all()

    def test_atom_array_with_inf_coords(self):
        """Test AtomArray with infinite coordinates - should remove those atoms."""
        atom_array = AtomArray(4)
        atom_array.coord = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, np.inf, 6.0],  # Has inf in y coordinate
                [7.0, 8.0, 9.0],
                [10.0, -np.inf, 12.0],  # Has -inf in y coordinate
            ]
        )
        atom_array.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        result = remove_atoms_with_any_nan_coords(atom_array)

        assert len(result) == 2
        assert list(cast(np.ndarray, result.res_id)) == [1, 3]
        assert np.isfinite(result.coord).all()

    def test_atom_array_all_nan(self):
        """Test AtomArray where all atoms have NaN - should return empty array."""
        atom_array = AtomArray(3)
        atom_array.coord = np.array(
            [
                [np.nan, 2.0, 3.0],
                [4.0, np.nan, 6.0],
                [7.0, 8.0, np.nan],
            ]
        )
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))

        result = remove_atoms_with_any_nan_coords(atom_array)

        assert len(result) == 0

    def test_atom_array_stack_all_finite(self):
        """Test AtomArrayStack with all finite coordinates."""
        # Create 2 models with 4 atoms each
        coords = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1], [10.1, 11.1, 12.1]],
            ]
        )

        atom_array1 = AtomArray(4)
        atom_array1.coord = coords[0]
        atom_array1.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array1.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array1.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_array2 = AtomArray(4)
        atom_array2.coord = coords[1]
        atom_array2.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array2.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array2.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_stack = stack([atom_array1, atom_array2])
        result = remove_atoms_with_any_nan_coords(atom_stack)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 2
        assert result.array_length() == 4
        # there's some float issue here, so we use almost equal.
        np.testing.assert_array_almost_equal(result.coord, coords)

    def test_atom_array_stack_nan_in_one_model(self):
        """Test AtomArrayStack where one model has NaN - should remove that atom from all models."""
        # Create 2 models, first has NaN in atom index 1
        coords = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1], [10.1, 11.1, 12.1]],
            ]
        )

        atom_array1 = AtomArray(4)
        atom_array1.coord = coords[0]
        atom_array1.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array1.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array1.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_array2 = AtomArray(4)
        atom_array2.coord = coords[1]
        atom_array2.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array2.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array2.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_stack = stack([atom_array1, atom_array2])
        result = remove_atoms_with_any_nan_coords(atom_stack)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 2
        assert result.array_length() == 3  # Atom at index 1 removed from all models
        assert list(cast(np.ndarray, result[0].res_id)) == [1, 3, 4]
        assert list(cast(np.ndarray, result[1].res_id)) == [1, 3, 4]
        assert np.isfinite(result.coord).all()

    def test_atom_array_stack_nan_in_different_models(self):
        """Test AtomArrayStack where different models have NaN in different atoms."""
        # Model 1 has NaN in atom 1, Model 2 has NaN in atom 2
        coords = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, np.nan, 9.1], [10.1, 11.1, 12.1]],
            ]
        )

        atom_array1 = AtomArray(4)
        atom_array1.coord = coords[0]
        atom_array1.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array1.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array1.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_array2 = AtomArray(4)
        atom_array2.coord = coords[1]
        atom_array2.set_annotation("chain_id", np.array(["A"] * 4))
        atom_array2.set_annotation("res_id", np.array([1, 2, 3, 4]))
        atom_array2.set_annotation("atom_name", np.array(["CA", "CB", "CG", "CD"]))

        atom_stack = stack([atom_array1, atom_array2])
        result = remove_atoms_with_any_nan_coords(atom_stack)

        # Both atoms 1 and 2 should be removed from all models
        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 2
        assert result.array_length() == 2
        assert list(cast(np.ndarray, result[0].res_id)) == [1, 4]
        assert list(cast(np.ndarray, result[1].res_id)) == [1, 4]
        assert np.isfinite(result.coord).all()

    def test_atom_array_stack_preserves_annotations(self):
        """Test that annotations are preserved after filtering."""
        coords = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
                [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            ]
        )

        atom_array1 = AtomArray(3)
        atom_array1.coord = coords[0]
        atom_array1.set_annotation("chain_id", np.array(["A", "A", "B"]))
        atom_array1.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array1.set_annotation("atom_name", np.array(["CA", "CB", "CG"]))
        atom_array1.set_annotation("res_name", np.array(["ALA", "VAL", "LEU"]))

        atom_array2 = AtomArray(3)
        atom_array2.coord = coords[1]
        atom_array2.set_annotation("chain_id", np.array(["A", "A", "B"]))
        atom_array2.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array2.set_annotation("atom_name", np.array(["CA", "CB", "CG"]))
        atom_array2.set_annotation("res_name", np.array(["ALA", "VAL", "LEU"]))

        atom_stack = stack([atom_array1, atom_array2])
        result = remove_atoms_with_any_nan_coords(atom_stack)

        assert list(cast(np.ndarray, result[0].chain_id)) == ["A", "B"]
        assert list(cast(np.ndarray, result[0].res_id)) == [1, 3]
        assert list(cast(np.ndarray, result[0].atom_name)) == ["CA", "CG"]
        assert list(cast(np.ndarray, result[0].res_name)) == ["ALA", "LEU"]

    def test_empty_atom_array(self):
        """Test with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.coord = np.empty((0, 3))

        with pytest.raises(ValueError, match="Cannot remove atoms from empty AtomArray\|Stack"):
            remove_atoms_with_any_nan_coords(atom_array)

    def test_empty_atom_array_stack(self):
        """Test with empty AtomArrayStack."""
        atom_array = AtomArray(0)
        atom_array.coord = np.empty((0, 3))
        atom_stack = stack([atom_array, atom_array])

        with pytest.raises(ValueError, match="Cannot remove atoms from empty AtomArray\|Stack"):
            remove_atoms_with_any_nan_coords(atom_stack)
