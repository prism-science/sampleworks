"""Coordinate-transform tests for the lunus.sf adapter.

These pin agreement between the three places a Cartesian-to-fractional matrix
comes from in this repo: qFit's closed-form ``UnitCell.orth_to_frac``, which the
real-space density path uses and which :func:`build_setup` now reads too; gemmi's
``fractionalization_matrix``, which SFC_Torch uses in the SFcalculator path; and
lunus's orthogonalization matrix, whose inverse the fractionalization matrix must
be. Nothing else checks this — qFit's own mutual-inverse assertion is commented
out in ``unitcell.py`` — and a silent divergence here would surface as an
unexplained disagreement when the engines are compared.

Cells cover the symmetry range that matters for the matrix: orthogonal, one
oblique angle, a 120 degree gamma, and all three angles oblique.
"""

import gemmi
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

# Imported below the guard, not above it: lunus_sf imports lunus.sf at module
# scope, so without lunus this has to skip rather than fail collection.
from sampleworks.core.forward_models.xray.lunus_sf import build_setup


CELLS = {
    "cubic": (30.0, 30.0, 30.0, 90.0, 90.0, 90.0),
    "monoclinic": (31.7, 42.3, 55.9, 90.0, 104.5, 90.0),
    "hexagonal": (61.2, 61.2, 98.4, 90.0, 90.0, 120.0),
    "triclinic": (23.1, 31.4, 44.7, 78.3, 85.1, 103.7),
}


@pytest.fixture
def atom_array() -> AtomArray:
    """Three atoms with the element and B-factor annotations build_setup needs."""
    atoms = AtomArray(3)
    atoms.coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    atoms.element = np.array(["C", "N", "O"])
    atoms.b_factor = np.array([20.0, 30.0, 15.0])
    return atoms


def setup_for(atom_array: AtomArray, cell_params: tuple[float, ...], **kwargs):
    """Build a setup for one cell in P1 at 2 A, float64 unless told otherwise."""
    kwargs.setdefault("dtype", torch.float64)
    return build_setup(atom_array, gemmi.UnitCell(*cell_params), "P 1", 2.0, **kwargs)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_frac_matrix_inverts_orth_matrix(atom_array, cell_params):
    """The two matrices must be mutual inverses. They come from different code —
    lunus builds one, qFit's closed form the other — so this is a real check."""
    setup = setup_for(atom_array, cell_params)
    product = setup.orth_matrix @ setup.frac_matrix
    torch.testing.assert_close(product, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_frac_matrix_matches_gemmi(atom_array, cell_params):
    """qFit's closed form must agree with gemmi's fractionalization matrix, the
    one SFC_Torch uses, so the two structure-factor engines fractionalize alike."""
    setup = setup_for(atom_array, cell_params)
    expected = np.array(gemmi.UnitCell(*cell_params).fractionalization_matrix.tolist())
    torch.testing.assert_close(setup.frac_matrix, torch.as_tensor(expected), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cell_params", CELLS.values(), ids=list(CELLS))
def test_fractionalized_coordinates_match_gemmi(atom_array, cell_params):
    """Applied to coordinates, including some outside the cell, the matrix must
    reproduce gemmi's own fractionalize()."""
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
    """The reason the numpy conversions cannot be reused: gradients have to reach
    the atomic coordinates through this transform."""
    setup = setup_for(atom_array, CELLS["triclinic"])
    coords = torch.tensor(atom_array.coord, dtype=torch.float64).unsqueeze(0)
    coords.requires_grad_(True)

    (coords @ setup.frac_matrix.T).sum().backward()

    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert not torch.allclose(coords.grad, torch.zeros_like(coords.grad))


def test_fractional_coordinates_are_not_wrapped(atom_array):
    """Coordinates outside the cell must stay outside it in fractional space.
    Wrapping into [0, 1) would put a gradient discontinuity at the cell boundary;
    lunus applies its own modulo later, in grid-index space."""
    setup = setup_for(atom_array, CELLS["monoclinic"])
    coords = torch.tensor([[[-12.0, -3.0, -40.0], [95.0, 130.0, 210.0]]], dtype=torch.float64)

    fractional = coords @ setup.frac_matrix.T

    assert (fractional[0, 0] < 0.0).all()
    assert (fractional[0, 1] > 1.0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cell_volume_matches_gemmi_at_either_dtype(atom_array, dtype):
    """The volume scales the structure factors, and is derived in float64 from the
    NumPy matrix, so it must not inherit float32 truncation from the kernels."""
    cell_params = CELLS["triclinic"]
    setup = setup_for(atom_array, cell_params, dtype=dtype)
    assert setup.cell_volume == pytest.approx(gemmi.UnitCell(*cell_params).volume, abs=1e-6)
