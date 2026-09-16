"""Tests for structure_utils module."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.utils.atom_array_utils import (
    apply_selection,
    map_altlocs_to_stack,
    parse_selection_string,
)
from sampleworks.utils.structure_utils import (
    _closest_canonical_amino_acid,
    canonicalize_mixed_altloc_residues,
    extract_selection_coordinates,
    get_asym_unit_from_structure,
    get_reference_atomarraystack,
    get_reference_structure_coords,
)


@pytest.fixture
def mock_protein_config(tmp_path: Path) -> ProteinConfig:
    """Mock ProteinConfig pointing to tmp_path for testing file operations."""
    return ProteinConfig(
        protein="test",
        base_map_dir=tmp_path,
        selection=[
            "chain A and resi 1-10",
        ],
        resolution=2.0,
        map_pattern="{occ_str}.ccp4",
        structure_pattern="{occ_str}.cif",
    )


class TestParseSelectionString:
    """Tests for parse_selection_string function."""

    def test_chain_only(self):
        """Test parsing chain-only selection."""
        result = parse_selection_string("chain A")
        assert result == ("A", None, None)

    def test_single_residue(self):
        """Test parsing single residue selection."""
        result = parse_selection_string("resi 10")
        assert result == (None, 10, 10)

    def test_residue_range(self):
        """Test parsing residue range selection."""
        result = parse_selection_string("resi 10-50")
        assert result == (None, 10, 50)

    def test_chain_and_residue_range(self):
        """Test parsing combined chain and residue range."""
        result = parse_selection_string("chain A and resi 10-50")
        assert result == ("A", 10, 50)

    def test_chain_and_single_residue(self):
        """Test parsing chain and single residue."""
        result = parse_selection_string("chain A and resi 10")
        assert result == ("A", 10, 10)

    def test_case_insensitive(self):
        """Test that parsing is case insensitive."""
        result = parse_selection_string("CHAIN a AND RESI 10")
        assert result == ("A", 10, 10)

    def test_empty_string(self, caplog):
        """Test parsing empty string returns warning and all None."""
        result = parse_selection_string("")
        assert result == (None, None, None)

        assert (
            "Selection string did not match any known patterns (e.g. 'chain A', 'resi 10-50')"
            in caplog.text
        )
        assert caplog.records[0].levelname == "WARNING"

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        result = parse_selection_string("chain  A  and  resi  10-20")
        assert result == ("A", 10, 20)


class TestApplySelection:
    """Tests for apply_selection function."""

    def test_none_selection_returns_unchanged(self, basic_atom_array_multichain):
        """Test that None selection returns original array."""
        result = apply_selection(basic_atom_array_multichain, None)
        assert result is basic_atom_array_multichain
        assert len(result) == len(basic_atom_array_multichain)

    def test_chain_selection(self, basic_atom_array_multichain):
        """Test filtering by chain."""
        result = apply_selection(basic_atom_array_multichain, "chain A")
        assert len(result) == 5
        assert all(result.chain_id == "A")

    def test_residue_selection(self, basic_atom_array_multichain):
        """Test filtering by residue range."""
        result = apply_selection(basic_atom_array_multichain, "resi 2-4")
        assert len(result) == 6
        res_ids = cast(np.ndarray, result.res_id)
        assert all((res_ids >= 2) & (res_ids <= 4))

    def test_single_residue_selection(self, basic_atom_array_multichain):
        """Test filtering by single residue."""
        result = apply_selection(basic_atom_array_multichain, "resi 3")
        assert len(result) == 2
        assert all(result.res_id == 3)

    def test_combined_selection(self, basic_atom_array_multichain):
        """Test filtering by chain and residue."""
        result = apply_selection(basic_atom_array_multichain, "chain A and resi 2-4")
        assert len(result) == 3
        assert all(result.chain_id == "A")
        res_ids = cast(np.ndarray, result.res_id)
        assert all((res_ids >= 2) & (res_ids <= 4))

    def test_no_matching_atoms_raises_valueerror(self, basic_atom_array_multichain):
        """Test that selection matching no atoms raises ValueError."""
        with pytest.raises(ValueError, match="matched no atoms"):
            apply_selection(basic_atom_array_multichain, "chain Z")

    def test_preserves_annotations(self, basic_atom_array_multichain):
        """Test that annotations are preserved after filtering."""
        result = apply_selection(basic_atom_array_multichain, "chain A")
        assert hasattr(result, "chain_id")
        assert hasattr(result, "res_id")
        assert hasattr(result, "atom_name")
        assert list(cast(np.ndarray, result.atom_name)) == ["CA"] * 5

    def test_preserves_coordinates(self, basic_atom_array_multichain):
        """Test that coordinates are preserved correctly."""
        result = apply_selection(basic_atom_array_multichain, "chain A and resi 1")
        result_coord = cast(np.ndarray, result.coord)
        basic_coord = cast(np.ndarray, basic_atom_array_multichain.coord)
        np.testing.assert_array_equal(result_coord[0], basic_coord[0])


class TestExtractSelectionCoordinates:
    """Tests for extract_selection_coordinates function."""

    def test_extracts_coordinates(self, basic_atom_array_multichain):
        """Test that coordinates are extracted correctly."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A")
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (5, 3)

    def test_with_atomarray_stack(self, atom_array_stack_simple):
        """Test extraction from AtomArrayStack uses first model."""
        coords = extract_selection_coordinates(atom_array_stack_simple, "resi 1-3")
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3, 3)
        first_model = cast(AtomArray, atom_array_stack_simple[0])
        np.testing.assert_array_equal(coords, first_model.coord[:3])

    def test_no_matching_atoms_raises_runtime_error(self, basic_atom_array_multichain):
        """Test that no matching atoms raises RuntimeError."""
        # Note: this is actually raising from get_mask_from_old_selection_string
        with pytest.raises(ValueError, match="Selection 'chain Z' matched no atoms"):
            extract_selection_coordinates(basic_atom_array_multichain, "chain Z")

        # this tests extract_selection_coordinates more directly
        with pytest.raises(RuntimeError, match="No atoms matched selection"):
            extract_selection_coordinates(basic_atom_array_multichain, "chain_id == 'Z'")

    def test_filters_nan_coordinates(self, caplog, atom_array_with_nan_coords):
        """Test that NaN coordinates are filtered out with warning."""
        coords = extract_selection_coordinates(atom_array_with_nan_coords, "chain A")
        assert len(coords) == 3
        assert np.isfinite(coords).all()

        assert "Filtered" in caplog.text
        assert "valid atoms remaining" in caplog.text
        assert "atoms with NaN/Inf coordinates" in caplog.text
        assert caplog.records[0].levelname == "WARNING"

    def test_all_nan_raises_runtime_error(self):
        """Test that all invalid coordinates raises RuntimeError."""
        atom_array = AtomArray(3)
        atom_array.coord = np.array([[np.nan, np.nan, np.nan]] * 3)
        atom_array.set_annotation("chain_id", np.array(["A", "A", "A"]))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No valid.*finite.*coordinates"):
            extract_selection_coordinates(atom_array, "chain A")

    def test_returns_numpy_array(self, basic_atom_array_multichain):
        """Test that output is numpy array."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A")
        assert isinstance(coords, np.ndarray)
        assert coords.dtype in (np.float32, np.float64)

    def test_coordinate_values(self, basic_atom_array_multichain):
        """Test that coordinate values are correct."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A and resi 1")
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(coords, expected)


class TestGetAsymUnitFromStructure:
    """Tests for get_asym_unit_from_structure function."""

    def test_returns_atomarray(self, basic_atom_array_multichain):
        """Test extraction of AtomArray from structure."""
        structure = {"asym_unit": basic_atom_array_multichain}
        result = get_asym_unit_from_structure(structure)
        assert isinstance(result, AtomArray)
        assert result is basic_atom_array_multichain

    def test_returns_atomarraystack(self, atom_array_stack_simple):
        """Test extraction of AtomArrayStack from structure."""
        structure = {"asym_unit": atom_array_stack_simple}
        result = get_asym_unit_from_structure(structure)
        assert isinstance(result, AtomArrayStack)
        assert result is atom_array_stack_simple

    def test_with_atom_array_index(self, atom_array_stack_simple):
        """Test extraction of specific model from stack."""
        structure = {"asym_unit": atom_array_stack_simple}
        result = get_asym_unit_from_structure(structure, atom_array_index=1)
        assert isinstance(result, AtomArray)
        assert len(result) == atom_array_stack_simple.array_length()

    def test_index_on_atomarray_ignored(self, basic_atom_array_multichain):
        """Test that index is ignored for AtomArray."""
        structure = {"asym_unit": basic_atom_array_multichain}
        result = get_asym_unit_from_structure(structure, atom_array_index=0)
        assert isinstance(result, AtomArray)
        assert result is basic_atom_array_multichain

    def test_invalid_type_raises_typeerror(self):
        """Test that invalid type raises TypeError."""
        structure = {"asym_unit": "not an atom array"}
        with pytest.raises(TypeError, match="Unexpected atom array type"):
            get_asym_unit_from_structure(structure)


class TestGetReferenceAtomArrayStack:
    """Tests for get_reference_atomarraystack function."""

    def test_returns_none_when_not_found(self, mock_protein_config):
        """Test that missing file returns (None, None)."""
        path, struct = get_reference_atomarraystack(mock_protein_config, {"A": 0.5, "B": 0.5})
        assert path is None
        assert struct is None

    def test_converts_atomarray_to_stack(self, tmp_path, basic_atom_array_multichain):
        """Test that single AtomArray is converted to stack."""
        from atomworks.io.utils.io_utils import to_cif_file

        structure_path = tmp_path / "0.5occA_0.5occB.cif"
        to_cif_file(basic_atom_array_multichain, structure_path)

        config = ProteinConfig(
            protein="test",
            base_map_dir=tmp_path,
            selection=[
                "chain A",
            ],
            resolution=2.0,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="{occ_str}.cif",
        )

        path, struct = get_reference_atomarraystack(config, {"A": 0.5, "B": 0.5})
        assert path is not None
        assert isinstance(struct, AtomArrayStack)
        assert struct.stack_depth() == 1

    def test_with_real_structure(self, resources_dir):
        """Test loading real structure with altlocs."""
        config = ProteinConfig(
            protein="6b8x",
            base_map_dir=resources_dir / "6b8x",
            selection=[
                "chain A",
            ],
            resolution=1.74,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="6b8x_final.pdb",
        )

        path, struct = get_reference_atomarraystack(config, {"A": 0.5, "B": 0.5})
        assert path is not None
        assert struct is not None
        assert isinstance(struct, AtomArrayStack)

    def test_6ni6_mixed_altloc_residue_stacks_cleanly(self, resources_dir):
        """End-to-end regression for the 6NI6 CYS/CSO/CSO compositional heterogeneity.

        Chain A residue 101 carries a canonical CYS (altloc A) alongside two CSO altlocs
        (B, C) with an extra ``OD`` atom. This tests failure via
        ``get_reference_atomarraystack`` -> ``map_altlocs_to_stack``.
        """
        config = ProteinConfig(
            protein="6ni6",
            base_map_dir=resources_dir / "6NI6",
            selection=[
                "chain A and resi 101",
            ],
            resolution=1.8,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="6NI6_single_001_density_input.cif",
        )

        _, ref_struct = get_reference_atomarraystack(config, {"A": 0.5, "B": 0.25, "C": 0.25})
        assert ref_struct is not None

        stacked, _ = map_altlocs_to_stack(
            ref_struct, selection="(chain_id == 'A') & (res_id == 101)", return_full_array=False
        )
        assert stacked.stack_depth() == 3  # altlocs A, B, C

        res_names = cast(np.ndarray, stacked.res_name)
        hetero = cast(np.ndarray, stacked.hetero)
        atom_names = cast(np.ndarray, stacked.atom_name)
        assert set(np.unique(res_names)) == {"CYS"}
        assert not hetero.any()
        assert "OD" not in atom_names  # CSO's incompatible extra atom is gone

        for i in range(stacked.stack_depth()):
            frame_atom_names = list(cast(np.ndarray, stacked[i].atom_name))
            assert len(frame_atom_names) == len(set(frame_atom_names))  # unique atom identities


class TestGetReferenceStructureCoords:
    """Tests for get_reference_structure_coords function."""

    def test_returns_none_when_no_valid(self, mock_protein_config):
        """Test that no valid structures returns None."""
        coords = get_reference_structure_coords(
            mock_protein_config, "test", occ_list=[{"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}]
        )
        assert coords is None

    def test_handles_exceptions_gracefully(self, tmp_path):
        """Test that exceptions are logged and function continues."""
        config = ProteinConfig(
            protein="test",
            base_map_dir=tmp_path,
            selection=[
                "chain Z and resi 999",
            ],
            resolution=2.0,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="{occ_str}.cif",
        )

        coords = get_reference_structure_coords(config, "test", occ_list=[{"A": 0.5, "B": 0.5}])
        assert coords is None

    def test_with_real_structure(self, resources_dir):
        """Test loading coords from real structure."""
        selection_string = "chain A and resi 1-10"
        config = ProteinConfig(
            protein="6b8x",
            base_map_dir=resources_dir / "6b8x",
            selection=[
                selection_string,
            ],
            resolution=1.74,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="6b8x_final.pdb",
        )

        coords_dict = get_reference_structure_coords(
            config, "6b8x", occ_list=[{"A": 0.5, "B": 0.5}]
        )
        assert coords_dict is not None
        assert isinstance(coords_dict, dict)
        assert len(coords_dict) == 1
        assert selection_string in coords_dict

        coords = coords_dict[selection_string]
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert np.isfinite(coords).all()


class TestCanonicalizeMixedAltlocResidues:
    """Tests for canonicalize_mixed_altloc_residues / _closest_canonical_amino_acid."""

    @staticmethod
    def _mixed_array() -> AtomArray:
        """Build a minimal array with one mixed and one untouched position.

        Returns
        -------
        AtomArray
            ``(A, 10)`` holds a canonical CYS plus a modified CSO (compositional
            heterogeneity); ``(A, 11)`` holds a lone MSE that must be left untouched.
        """
        aa = AtomArray(3)
        aa.coord = np.zeros((3, 3), dtype=np.float32)
        aa.set_annotation("chain_id", np.array(["A", "A", "A"]))
        aa.set_annotation("res_id", np.array([10, 10, 11]))
        aa.set_annotation("res_name", np.array(["CYS", "CSO", "MSE"]))
        aa.set_annotation("hetero", np.array([False, True, True]))
        aa.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))
        return aa

    @staticmethod
    def _mixed_array_with_altlocs() -> AtomArray:
        """Build a mixed CYS/CSO position with per-altloc atoms and an extra CSO atom.

        Returns
        -------
        AtomArray
            ``(A, 10)`` as canonical CYS (altloc ``A``: ``CA``, ``SG``) alternating with a
            modified CSO (altloc ``B``: ``CA``, ``SG`` and the incompatible extra ``OD``).
        """
        aa = AtomArray(5)
        aa.coord = np.zeros((5, 3), dtype=np.float32)
        aa.set_annotation("chain_id", np.array(["A", "A", "A", "A", "A"]))
        aa.set_annotation("res_id", np.array([10, 10, 10, 10, 10]))
        aa.set_annotation("res_name", np.array(["CYS", "CYS", "CSO", "CSO", "CSO"]))
        aa.set_annotation("hetero", np.array([False, False, True, True, True]))
        aa.set_annotation("atom_name", np.array(["CA", "SG", "CA", "SG", "OD"]))
        aa.set_annotation("altloc_id", np.array(["A", "A", "B", "B", "B"]))
        return aa

    @pytest.mark.parametrize(
        "res_name,expected",
        [("CSO", "CYS"), ("MSE", "MET"), ("ALA", "ALA"), ("HOH", None)],
    )
    def test_closest_canonical_amino_acid(self, res_name, expected):
        """The modified-to-canonical mapping resolves PTMs and rejects non-amino-acids."""
        assert _closest_canonical_amino_acid(res_name) == expected

    def test_renames_modified_at_mixed_position(self):
        """Modified records at a mixed position are renamed and de-heteroed; lone MSE is kept."""
        result = canonicalize_mixed_altloc_residues(self._mixed_array())
        res_name = cast(np.ndarray, result.res_name)
        hetero = cast(np.ndarray, result.hetero)
        assert list(res_name[:2]) == ["CYS", "CYS"]  # CSO -> CYS
        assert not hetero[0] and not hetero[1]  # hetero cleared
        assert res_name[2] == "MSE" and hetero[2]  # MSE untouched

    def test_drops_incompatible_modified_atoms(self):
        """The modified form's extra atom is dropped so both altlocs share one atom set."""
        result = canonicalize_mixed_altloc_residues(self._mixed_array_with_altlocs())
        atom_name = cast(np.ndarray, result.atom_name)
        altloc_id = cast(np.ndarray, result.altloc_id)
        assert "OD" not in set(atom_name)  # incompatible extra removed
        assert set(atom_name[altloc_id == "A"]) == set(atom_name[altloc_id == "B"])
        assert set(cast(np.ndarray, result.res_name)) == {"CYS"}
        assert not cast(np.ndarray, result.hetero).any()

    def test_two_noncanonicals_no_canonical_warns(self, caplog):
        """Two non-canonicals with no shared canonical parent are left untouched and logged."""
        aa = AtomArray(2)
        aa.coord = np.zeros((2, 3), dtype=np.float32)
        aa.set_annotation("chain_id", np.array(["A", "A"]))
        aa.set_annotation("res_id", np.array([20, 20]))
        aa.set_annotation("res_name", np.array(["CSO", "SEP"]))
        aa.set_annotation("hetero", np.array([True, True]))
        aa.set_annotation("atom_name", np.array(["CA", "CA"]))
        aa.set_annotation("altloc_id", np.array(["A", "B"]))
        with caplog.at_level("WARNING"):
            result = canonicalize_mixed_altloc_residues(aa)
        # CSO -> CYS and SEP -> SER don't converge, so neither is renamed (a partial rename
        # would still leave the position stacking-incompatible).
        assert set(cast(np.ndarray, result.res_name)) == {"CSO", "SEP"}
        assert cast(np.ndarray, result.hetero).all()
        assert "canonical parent" in caplog.text

    def test_partial_convergence_leaves_position_untouched(self, caplog):
        """A canonical plus two non-convergent modified forms is left fully untouched."""
        aa = AtomArray(3)
        aa.coord = np.zeros((3, 3), dtype=np.float32)
        aa.set_annotation("chain_id", np.array(["A", "A", "A"]))
        aa.set_annotation("res_id", np.array([30, 30, 30]))
        aa.set_annotation("res_name", np.array(["CYS", "CSO", "SEP"]))
        aa.set_annotation("hetero", np.array([False, True, True]))
        aa.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))
        aa.set_annotation("altloc_id", np.array(["A", "B", "C"]))
        with caplog.at_level("WARNING"):
            result = canonicalize_mixed_altloc_residues(aa)
        assert list(cast(np.ndarray, result.res_name)) == ["CYS", "CSO", "SEP"]
        assert "canonical parent" in caplog.text

    def test_does_not_mutate_input(self):
        """Canonicalization operates on a copy and leaves the caller's array intact."""
        arr = self._mixed_array()
        canonicalize_mixed_altloc_residues(arr)
        assert cast(np.ndarray, arr.res_name)[1] == "CSO"
