"""Density maps from the PyTorch dilation path compared with independent references: the
analytic Gaussian-sum density of a single atom, and gemmi's density calculators."""

import gemmi
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    GridParameters,
    Resolution,
    XMap,
)
from sampleworks.core.losses.real_space_density import setup_scattering_params
from sampleworks.utils.density_utils import build_density_transformer, run_density_transformer


CPU = torch.device("cpu")
LOW_RESOLUTION = 1000.0
# Default DensityParameters.rmax: the transformer writes no density farther than this from an atom.
DENSITY_RADIUS = 3.0


def orthorhombic_p1_grid(origin: np.ndarray, voxel_counts: np.ndarray, resolution: float) -> XMap:
    """An empty P1 map whose cell spans exactly ``voxel_counts`` voxels of ``resolution / 4``.

    Parameters
    ----------
    origin
        Cartesian position of voxel (0, 0, 0), shape (3,).
    voxel_counts
        Voxels along x, y and z, shape (3,).
    resolution
        High-resolution limit in Angstroms.

    Returns
    -------
    XMap
        Zero-filled map with array shape (nz, ny, nx).
    """
    voxel_spacing = resolution / 4
    a, b, c = (voxel_counts * voxel_spacing).tolist()
    return XMap(
        np.zeros(voxel_counts[::-1], dtype=np.float32),
        grid_parameters=GridParameters(voxelspacing=voxel_spacing),
        unit_cell=UnitCell(a=a, b=b, c=c, alpha=90.0, beta=90.0, gamma=90.0, space_group="P1"),
        resolution=Resolution(high=resolution, low=LOW_RESOLUTION),
        origin=origin,
    )


def sampleworks_density(
    atom_array: AtomArray, xmap: XMap, em_mode: bool, scattering_table: str
) -> np.ndarray:
    """Density of ``atom_array`` on ``xmap``'s grid from the PyTorch dilation path.

    Parameters
    ----------
    atom_array
        Atoms with ``b_factor`` and ``occupancy`` annotations.
    xmap
        Grid to compute on.
    em_mode
        Whether to use electron scattering factors.
    scattering_table
        Scattering table name passed to ``setup_scattering_params``.

    Returns
    -------
    np.ndarray
        Density of shape (nz, ny, nx).
    """
    transformer, _ = build_density_transformer(
        xmap, em_mode=em_mode, device=CPU, scattering_table=scattering_table
    )
    return run_density_transformer(transformer, atom_array).numpy()


def voxel_coordinates(xmap: XMap) -> np.ndarray:
    """Cartesian position of every voxel of an orthorhombic map.

    Parameters
    ----------
    xmap
        Map with an orthorhombic cell.

    Returns
    -------
    np.ndarray
        Positions of shape (nz, ny, nx, 3), in (x, y, z) order.
    """
    axes = [
        xmap.origin[axis] + np.arange(xmap.array.shape[2 - axis]) * xmap.voxelspacing[axis]
        for axis in range(3)
    ]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return np.stack([x, y, z], axis=-1)


def gaussian_sum_density(
    distances: np.ndarray, coefficients: np.ndarray, b_factor: float, em_mode: bool
) -> np.ndarray:
    """Real-space density of one atom with no resolution limit, in electrons per cubic Angstrom.

    Each term ``a * exp(-b s^2)`` of the scattering factor, blurred by the B-factor, is the
    Fourier transform of ``a (4 pi / (b + B))^1.5 exp(-4 pi^2 r^2 / (b + B))``.

    Parameters
    ----------
    distances
        Distances from the atom in Angstroms, any shape.
    coefficients
        One element's row of ``setup_scattering_params``, shape (n_coeffs, 2) of (a, b)
        pairs; for X-rays, row 5 holds the constant term in its ``a`` column.
    b_factor
        Isotropic B-factor in square Angstroms.
    em_mode
        Whether ``coefficients`` are electron scattering factors, which have no constant term.

    Returns
    -------
    np.ndarray
        Density with the shape of ``distances``.
    """
    amplitudes, widths = coefficients[:5, 0], coefficients[:5, 1] + b_factor
    if not em_mode:
        amplitudes = np.append(amplitudes, coefficients[5, 0])
        widths = np.append(widths, b_factor)
    density = np.zeros_like(distances)
    for amplitude, width in zip(amplitudes, widths):
        density += (
            amplitude * (4 * np.pi / width) ** 1.5 * np.exp(-4 * np.pi**2 * distances**2 / width)
        )
    return density


def gemmi_density(atom_array: AtomArray, xmap: XMap, em_mode: bool) -> np.ndarray:
    """Density of ``atom_array`` from gemmi, which uses the IT92 (X-ray) or C4322 (electron)
    tables and no resolution limit.

    Parameters
    ----------
    atom_array
        Atoms with ``b_factor`` and ``occupancy`` annotations.
    xmap
        Orthorhombic P1 map whose cell spans its grid exactly.
    em_mode
        Whether to use ``DensityCalculatorE`` rather than ``DensityCalculatorX``.

    Returns
    -------
    np.ndarray
        Density of shape (nz, ny, nx).
    """
    calculator = gemmi.DensityCalculatorE() if em_mode else gemmi.DensityCalculatorX()
    calculator.d_min = xmap.resolution.high
    calculator.blur = 0.0
    calculator.grid.unit_cell = gemmi.UnitCell(*xmap.unit_cell.abc.tolist(), 90, 90, 90)
    calculator.grid.spacegroup = gemmi.SpaceGroup("P1")
    calculator.grid.set_size(*xmap.array.shape[::-1])
    # gemmi's grid starts at the cell origin, so atoms are placed relative to the map origin.
    for position, element, b_factor, occupancy in zip(
        atom_array.coord - xmap.origin,
        atom_array.element,
        atom_array.b_factor,
        atom_array.occupancy,
    ):
        atom = gemmi.Atom()
        atom.pos = gemmi.Position(*position.tolist())
        atom.element = gemmi.Element(element)
        atom.b_iso = float(b_factor)
        atom.occ = float(occupancy)
        calculator.add_atom_density_to_grid(atom)
    return np.array(calculator.grid, copy=True).transpose(2, 1, 0)


def band_limit(density: np.ndarray, voxel_spacing: float, resolution: float) -> np.ndarray:
    """Keep only the Fourier components of ``density`` between ``LOW_RESOLUTION`` and
    ``resolution``, the band the sampleworks transformer integrates over.

    Parameters
    ----------
    density
        Periodic density on a cubic-voxel grid, shape (nz, ny, nx).
    voxel_spacing
        Voxel edge length in Angstroms.
    resolution
        High-resolution limit in Angstroms.

    Returns
    -------
    np.ndarray
        Band-limited density with the shape of ``density``.
    """
    structure_factors = np.fft.fftn(density)
    frequencies = np.meshgrid(
        *[np.fft.fftfreq(n, d=voxel_spacing) for n in density.shape], indexing="ij"
    )
    inverse_d = np.sqrt(sum(frequency**2 for frequency in frequencies))
    structure_factors[(inverse_d > 1 / resolution) | (inverse_d < 1 / LOW_RESOLUTION)] = 0
    return np.fft.ifftn(structure_factors).real


def correlation_and_scale(predicted: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Pearson correlation of two maps, and the least-squares scale taking ``reference``
    onto ``predicted`` after both are mean-centered.

    Parameters
    ----------
    predicted
        Map under test.
    reference
        Reference map of the same shape.

    Returns
    -------
    tuple[float, float]
        Correlation and scale.
    """
    predicted_centered = predicted.ravel() - predicted.mean()
    reference_centered = reference.ravel() - reference.mean()
    covariance = predicted_centered @ reference_centered
    correlation = covariance / np.sqrt(
        (predicted_centered @ predicted_centered) * (reference_centered @ reference_centered)
    )
    return float(correlation), float(covariance / (reference_centered @ reference_centered))


def single_atom(element: str, position: np.ndarray, b_factor: float) -> AtomArray:
    """A one-atom structure at full occupancy.

    Parameters
    ----------
    element
        Element symbol.
    position
        Cartesian position, shape (3,).
    b_factor
        Isotropic B-factor in square Angstroms.

    Returns
    -------
    AtomArray
        The atom, with ``b_factor`` and ``occupancy`` annotations.
    """
    atom_array = AtomArray(1)
    atom_array.coord[0] = position
    atom_array.element[0] = element
    atom_array.add_annotation("b_factor", float)
    atom_array.add_annotation("occupancy", float)
    atom_array.b_factor[0] = b_factor
    atom_array.occupancy[0] = 1.0
    return atom_array


@pytest.fixture(scope="module")
def peptide_1vme(structure_1vme: dict) -> AtomArray:
    """The modeled atoms of the first 19 residues of 1VME chain A, which include
    selenomethionine and half-occupancy atoms."""
    atom_array = structure_1vme["asym_unit"]
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    # atomworks keeps unresolved atoms with NaN coordinates.
    modeled = np.isfinite(atom_array.coord).all(axis=-1)
    return atom_array[modeled & (atom_array.chain_id == "A") & (atom_array.res_id < 20)]


@pytest.mark.parametrize(
    ("em_mode", "scattering_table"), [(False, "it1992"), (True, "c4322")], ids=["xray", "em"]
)
def test_single_atom_matches_the_analytic_gaussian_sum(em_mode, scattering_table):
    """At 0.5 A, where the transform of a B=20 carbon has died out, the band-limited density
    of an off-grid atom on a non-cubic grid equals the unlimited Gaussian-sum density."""
    resolution, b_factor = 0.5, 20.0
    xmap = orthorhombic_p1_grid(
        origin=np.array([-3.0, 2.0, 7.0]),
        voxel_counts=np.array([80, 96, 112]),
        resolution=resolution,
    )
    position = xmap.origin + np.array([4.03, 6.17, 7.71])

    predicted = sampleworks_density(
        single_atom("C", position, b_factor), xmap, em_mode, scattering_table
    )

    carbon = setup_scattering_params(
        em_mode=em_mode, device=CPU, scattering_table=scattering_table
    )[6].numpy()
    distances = np.linalg.norm(voxel_coordinates(xmap) - position, axis=-1)
    expected = gaussian_sum_density(distances, carbon, b_factor, em_mode)
    # A voxel-width margin keeps voxels at the cutoff from depending on float rounding.
    within_radius = distances < DENSITY_RADIUS - xmap.voxelspacing[0]
    np.testing.assert_allclose(
        predicted[within_radius], expected[within_radius], rtol=0, atol=1e-4 * expected.max()
    )


@pytest.mark.parametrize("resolution", [1.2, 2.0])
@pytest.mark.parametrize(
    ("em_mode", "scattering_table"), [(False, "it1992"), (True, "c4322")], ids=["xray", "em"]
)
def test_peptide_matches_band_limited_gemmi_density(
    peptide_1vme, em_mode, scattering_table, resolution
):
    """With the same scattering table, a peptide's density matches gemmi's once gemmi's map
    is limited to the same resolution. The scale is a little low at 2 A because the
    transformer cuts each atom off at ``DENSITY_RADIUS``, dropping the outer ripples a
    resolution limit puts in the atomic density."""
    padding = 6.0
    origin = peptide_1vme.coord.min(axis=0) - padding
    extent = peptide_1vme.coord.max(axis=0) + padding - origin
    xmap = orthorhombic_p1_grid(origin, np.ceil(extent / (resolution / 4)).astype(int), resolution)

    predicted = sampleworks_density(peptide_1vme, xmap, em_mode, scattering_table)
    reference = band_limit(
        gemmi_density(peptide_1vme, xmap, em_mode), xmap.voxelspacing[0], resolution
    )

    correlation, scale = correlation_and_scale(predicted, reference)
    assert correlation > 0.997
    assert scale == pytest.approx(1.0, abs=0.02)
