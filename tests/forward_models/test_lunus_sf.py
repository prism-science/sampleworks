"""Tests for the lunus.sf adapter.

Coordinate transforms are checked against the other two sources of a
fractionalization matrix in the repo, qFit's ``UnitCell.orth_to_frac`` and gemmi's
``fractionalization_matrix`` (used by SFC_Torch), over orthogonal, monoclinic,
hexagonal and triclinic cells. The blur and the symmetry operations are checked
against the values the module documents.
"""

import gemmi
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Below the guard: lunus_sf imports lunus.sf at module scope.
from lunus.sf.cell_utils import recommended_blur
from sampleworks.core.forward_models.xray.lunus_sf import (
    build_setup,
    space_group_operations,
    structure_factors,
)


CELLS = {
    "cubic": (30.0, 30.0, 30.0, 90.0, 90.0, 90.0),
    "monoclinic": (31.7, 42.3, 55.9, 90.0, 104.5, 90.0),
    "hexagonal": (61.2, 61.2, 98.4, 90.0, 90.0, 120.0),
    "triclinic": (23.1, 31.4, 44.7, 78.3, 85.1, 103.7),
}


@pytest.fixture
def atom_array() -> AtomArray:
    """Three atoms with the annotations build_setup reads."""
    atoms = AtomArray(3)
    atoms.coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    atoms.element = np.array(["C", "N", "O"])
    atoms.b_factor = np.array([20.0, 30.0, 15.0])
    return atoms


def setup_for(atom_array: AtomArray, cell_params: tuple[float, ...], **kwargs):
    """Setup for one cell in P1 at 2 A, float64 by default."""
    kwargs.setdefault("dtype", torch.float64)
    return build_setup(atom_array, gemmi.UnitCell(*cell_params), "P 1", 2.0, **kwargs)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_frac_matrix_inverts_orth_matrix(atom_array, cell_params):
    """lunus's orthogonalization matrix and qFit's fractionalization matrix are inverses."""
    setup = setup_for(atom_array, cell_params)
    product = setup.orth_matrix @ setup.frac_matrix
    torch.testing.assert_close(product, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_frac_matrix_matches_gemmi(atom_array, cell_params):
    """qFit's fractionalization matrix equals gemmi's, the one SFC_Torch uses."""
    setup = setup_for(atom_array, cell_params)
    expected = np.array(gemmi.UnitCell(*cell_params).fractionalization_matrix.tolist())
    torch.testing.assert_close(setup.frac_matrix, torch.as_tensor(expected), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_fractionalized_coordinates_match_gemmi(atom_array, cell_params):
    """The matrix reproduces gemmi's fractionalize(), including outside the cell."""
    setup = setup_for(atom_array, cell_params)
    coords = torch.tensor(
        [[[1.3, -4.7, 22.9], [15.0, 8.2, -3.1], [88.4, 71.6, 130.2]]], dtype=torch.float64
    )
    fractional = coords @ setup.frac_matrix.T

    cell = gemmi.UnitCell(*cell_params)
    expected = np.stack(
        [np.array(cell.fractionalize(gemmi.Position(*xyz)).tolist()) for xyz in coords[0].numpy()]
    )
    torch.testing.assert_close(fractional[0], torch.as_tensor(expected), atol=1e-12, rtol=0)


def test_fractionalization_is_differentiable(atom_array):
    """Gradients reach the Cartesian coordinates through the transform."""
    setup = setup_for(atom_array, CELLS["triclinic"])
    coords = torch.tensor(atom_array.coord, dtype=torch.float64).unsqueeze(0)
    coords.requires_grad_(True)

    (coords @ setup.frac_matrix.T).sum().backward()

    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert not torch.allclose(coords.grad, torch.zeros_like(coords.grad))


def test_fractional_coordinates_are_not_wrapped(atom_array):
    """Coordinates outside the cell stay outside it in fractional space; lunus
    applies its own modulo in grid-index space."""
    setup = setup_for(atom_array, CELLS["monoclinic"])
    coords = torch.tensor([[[-12.0, -3.0, -40.0], [95.0, 130.0, 210.0]]], dtype=torch.float64)

    fractional = coords @ setup.frac_matrix.T

    assert (fractional[0, 0] < 0.0).all()
    assert (fractional[0, 1] > 1.0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cell_volume_matches_gemmi_at_either_dtype(atom_array, dtype):
    """The volume is computed in float64 regardless of the kernels' dtype."""
    cell_params = CELLS["triclinic"]
    setup = setup_for(atom_array, cell_params, dtype=dtype)
    assert setup.cell_volume == pytest.approx(gemmi.UnitCell(*cell_params).volume, abs=1e-6)


class TestRecommendedBlur:
    """build_setup applies lunus's recommended_blur, sized to the grid it built."""

    def test_build_setup_applies_it_by_default(self, atom_array):
        """``blur=None`` takes lunus's recommendation for the sharpest atom; ``0.0`` disables it."""
        sharp = atom_array.copy()
        sharp.b_factor = np.full(sharp.array_length(), 2.0)

        auto = setup_for(sharp, CELLS["monoclinic"])
        explicit_off = setup_for(sharp, CELLS["monoclinic"], blur=0.0)

        expected = recommended_blur(2.0, auto.grid_shape, auto.orth_matrix.numpy(), d_min=2.0)
        assert auto.blur == pytest.approx(expected)
        assert auto.blur > 0.0
        assert explicit_off.blur == 0.0

    def test_is_zero_when_every_atom_is_already_smooth(self, atom_array):
        smooth = atom_array.copy()
        smooth.b_factor = np.full(smooth.array_length(), 90.0)

        assert setup_for(smooth, CELLS["monoclinic"]).blur == 0.0


class TestSymmetryOperations:
    """Symmetry operations read from gemmi, and the grid sized to accommodate them.

    gemmi splits a group into ``sym_ops`` and ``cen_ops``; reading only the former
    undercounts a centred group by the centring multiplicity (48 instead of 192 for
    F m -3 m). ``operations()`` yields the full set.
    """

    # (Hermann-Mauguin symbol, general-position multiplicity): primitive, the
    # three centrings, the largest groups, and the quarter-translation cases.
    GROUPS = {
        "P1": ("P 1", 1),
        "monoclinic screw": ("P 1 21 1", 2),
        "trigonal": ("P 31 2 1", 6),
        "C-centred": ("C 1 2 1", 4),
        "I-centred": ("I 4", 8),
        "F-centred cubic": ("F 2 3", 48),
        "I-centred cubic": ("I a -3 d", 96),
        "largest": ("F m -3 m", 192),
        "diamond glide": ("F d -3 c", 192),
    }

    @pytest.fixture
    @staticmethod
    def cubic_cell() -> gemmi.UnitCell:
        """A cubic cell, compatible with every group in GROUPS."""
        return gemmi.UnitCell(120.0, 120.0, 120.0, 90.0, 90.0, 90.0)

    @pytest.mark.parametrize("hm,multiplicity", GROUPS.values(), ids=list(GROUPS))
    def test_operation_count_matches_the_group_order(self, hm, multiplicity):
        """The operation count is the group order, centring included."""
        rotations, translations = space_group_operations(hm)

        assert len(rotations) == multiplicity
        assert len(translations) == multiplicity

    @pytest.mark.parametrize("hm,multiplicity", GROUPS.values(), ids=list(GROUPS))
    def test_grid_ops_drop_only_the_identity(self, atom_array, cubic_cell, hm, multiplicity):
        """``build_grid_ops`` removes only the identity; empty for P1."""
        setup = build_setup(atom_array, cubic_cell, hm, 4.0)

        assert len(setup.grid_ops) == multiplicity - 1

    @pytest.mark.parametrize("hm,multiplicity", GROUPS.values(), ids=list(GROUPS))
    def test_grid_is_commensurate_with_every_translation(
        self, atom_array, cubic_cell, hm, multiplicity
    ):
        """Every fractional translation lands on a grid point. Half translations
        force an even axis; the quarter translations of F d -3 c force a multiple
        of four (96 where F m -3 m gets 90)."""
        setup = build_setup(atom_array, cubic_cell, hm, 4.0)
        _, translations = space_group_operations(hm)
        grid = np.array(setup.grid_shape, dtype=np.float64)

        for translation in translations:
            on_grid = translation * grid
            assert np.allclose(on_grid, np.rint(on_grid), atol=1e-6), (
                f"{hm}: translation {translation} does not land on grid {tuple(grid)}"
            )

    def test_forward_pass_runs_under_the_largest_group(self, atom_array, cubic_cell):
        """The expansion path runs under F m -3 m, 191 non-identity operations."""
        setup = build_setup(atom_array, cubic_cell, "F m -3 m", 6.0)
        coords = torch.as_tensor(atom_array.coord[None], dtype=torch.float32)
        occupancies = torch.ones(atom_array.array_length())
        hkl = torch.tensor([[1, 1, 1], [2, 0, 0], [2, 2, 0]], dtype=torch.long)

        amplitudes = structure_factors(setup, coords, occupancies, hkl)

        assert amplitudes.shape == (1, len(hkl))
        assert torch.isfinite(amplitudes).all()
        assert torch.abs(amplitudes).max() > 0, "expansion produced no density"
