from typing import Any, cast

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack

from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    GridParameters,
    Resolution,
    XMap,
)
from sampleworks.core.rewards.real_space_density import (
    extract_density_inputs_from_atomarray,
    setup_scattering_params,
)


def create_synthetic_grid(
    atom_array: AtomArray | AtomArrayStack, resolution: float, padding: float = 5.0
) -> XMap:
    """Create an empty density map grid sized to fit the structure.

    Parameters
    ----------
    atom_array
        Structure to create a grid for
    resolution
        Map resolution in Angstroms
    padding
        Extra space to add around the structure in each dimension (Angstroms)

    Returns
    -------
    XMap
        Empty density map with appropriate grid parameters and unit cell
        dimensions to contain the structure
    """
    coords = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.coord)
    if coords.ndim == 3:
        coords = coords.reshape(-1, 3)

    valid_mask = np.isfinite(coords).all(axis=1)
    coords = coords[valid_mask]

    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding
    extent = max_coords - min_coords

    # standard voxel spacing from Phenix, etc.
    voxel_spacing = resolution / 4.0
    grid_shape = np.ceil(extent / voxel_spacing).astype(int)

    unit_cell = UnitCell(
        a=float(extent[0]),
        b=float(extent[1]),
        c=float(extent[2]),
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        space_group="P1",
    )

    # (nz, ny, nx) ordering for array, since most fft routines will expect
    # z as the "fastest" axis, but CCP4 format uses X as the fastest axis
    empty_array = np.zeros((grid_shape[2], grid_shape[1], grid_shape[0]), dtype=np.float32)

    empty_xmap = XMap(
        empty_array,
        grid_parameters=GridParameters(voxelspacing=voxel_spacing),
        unit_cell=unit_cell,
        resolution=Resolution(high=resolution, low=1000.0),
        origin=min_coords,
    )

    return empty_xmap


def build_density_transformer(
    xmap: XMap,
    em_mode: bool,
    device: torch.device,
    scattering_table: str = "wk1995",
) -> tuple[DifferentiableTransformer, XMap_torch]:
    """Build a reusable DifferentiableTransformer bound to a grid.

    Callers that compute density for many structures on the same grid can
    build the transformer once and reuse it across calls, avoiding the cost
    of GaussLegendreQuadrature setup and transform-matrix initialization
    inside :class:`DifferentiableTransformer.__init__`.

    Parameters
    ----------
    xmap : XMap
        Map whose grid geometry (unit cell, voxel spacing, resolution,
        origin) defines the output density grid.
    em_mode : bool
        If True, use electron scattering factors. If False, use X-ray factors.
    device : torch.device
        PyTorch device for computation.
    scattering_table : str
        X-ray scattering table, ``"wk1995"`` (default) or ``"it1992"``.

    Returns
    -------
    tuple[DifferentiableTransformer, XMap_torch]
        The transformer and the XMap_torch it was built against.
    """
    scattering_params = setup_scattering_params(
        em_mode=em_mode, device=device, scattering_table=scattering_table
    )

    xmap_torch = XMap_torch(xmap, device=device)
    transformer = DifferentiableTransformer(
        xmap=xmap_torch,
        scattering_params=scattering_params,
        em=em_mode,
        device=device,
        use_cuda_kernels=device.type == "cuda" and torch.cuda.is_available(),
    )
    return transformer, xmap_torch


def run_density_transformer(
    transformer: DifferentiableTransformer,
    atom_array: AtomArray | AtomArrayStack,
) -> torch.Tensor:
    """Run a prebuilt density transformer on an atom array.

    For :class:`~biotite.structure.AtomArrayStack` inputs, density is computed
    independently for each model's coordinates and then summed.

    Parameters
    ----------
    transformer : DifferentiableTransformer
        Prebuilt transformer (see :func:`build_density_transformer`). Inputs
        are placed on ``transformer.device`` and the returned tensor is on the
        same device.
    atom_array : AtomArray | AtomArrayStack
        Structure to compute density for.

    Returns
    -------
    torch.Tensor
        Computed electron density on the transformer's grid.
    """
    coords, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
        atom_array, transformer.device
    )

    with torch.no_grad():
        density = transformer(
            coordinates=coords,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    return density.sum(dim=0)


def compute_density_from_atomarray(
    atom_array: AtomArray | AtomArrayStack,
    xmap: XMap | None = None,
    resolution: float | None = None,
    em_mode: bool = False,
    device: torch.device | None = None,
    scattering_table: str = "wk1995",
) -> tuple[torch.Tensor, XMap_torch]:
    """Compute synthetic electron density from atomic coordinates.

    Either xmap OR resolution must be provided:
    - If xmap is provided: uses existing grid parameters (for RSCC evaluation)
    - If resolution is provided: creates synthetic grid (for density generation)

    For :class:`~biotite.structure.AtomArrayStack` inputs, density is computed
    independently for each model's coordinates and then summed. Occupancy
    values are taken directly from the structure annotations (shared across
    models in :class:`~biotite.structure.AtomArrayStack`).

    For callers that repeatedly compute density on the same grid, use
    :func:`build_density_transformer` and :func:`run_density_transformer`
    directly to reuse the expensive transformer setup.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to compute density for
    xmap : XMap | None
        Optional XMap with existing grid parameters. If provided, resolution is ignored.
    resolution : float | None
        Map resolution in Angstroms. Used to create synthetic grid if xmap is not provided.
    em_mode : bool
        If True, use electron scattering factors. If False, use X-ray factors.
    device : torch.device | None
        PyTorch device for computation. If None, will try to use GPU.
    scattering_table : str
        X-ray scattering table, ``"wk1995"`` (default) or ``"it1992"``.

    Returns
    -------
    tuple[torch.Tensor, XMap_torch]
        Tuple of (density tensor, XMap_torch object). The density tensor contains
        the computed electron density values on the grid.

    Raises
    ------
    ValueError
        If neither xmap nor resolution is provided, or if both are provided.
    """
    if device is None:
        from sampleworks.utils.torch_utils import try_gpu

        device = try_gpu()

    if xmap is None and resolution is None:
        raise ValueError("Either xmap or resolution must be provided")
    if xmap is not None and resolution is not None:
        raise ValueError("Cannot provide both xmap and resolution; choose one")

    if xmap is None:
        xmap = create_synthetic_grid(atom_array, resolution, padding=5.0)  # ty: ignore[invalid-argument-type] (resolution will not be None here)

    transformer, xmap_torch = build_density_transformer(
        xmap, em_mode=em_mode, device=device, scattering_table=scattering_table
    )
    density = run_density_transformer(transformer, atom_array)
    return density, xmap_torch
