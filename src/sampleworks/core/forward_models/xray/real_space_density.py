# ty: ignore
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from .real_space_density_deps.ops.dilate_points_cuda import dilate_atom_centric
from .real_space_density_deps.qfit.spacegroups import GetSpaceGroup
from .real_space_density_deps.qfit.unitcell import UnitCell
from .real_space_density_deps.qfit.volume import GridParameters, Resolution, XMap
from .real_space_density_deps.utils.interpolation import tricubic_interpolation_torch
from .real_space_density_deps.utils.quadrature import GaussLegendreQuadrature


@dataclass
class DensityParameters:
    """Parameters for electron density calculation.

    Controls the numerical parameters used in density calculations and
    integration procedures.

    Parameters
    ----------
    rmax: float
        Maximum radius for density calculation in Angstroms.
    rstep: float
        Step size for radial grid in Angstroms.
    smin: float
        Minimum scattering vector magnitude in inverse Angstroms.
    smax: float
        Maximum scattering vector magnitude in inverse Angstroms. Default is based
        on 1.0 Å.
    quad_points: int
        Number of quadrature points for numerical integration.
    integration_method: str
        Integration method to use ('gausslegendre' or 'simpson' or 'trapezoid').
    """

    rmax: float = 3.0
    rstep: float = 0.01
    smin: float = 0.0
    smax: float = 0.5
    quad_points: int = 50
    integration_method: str = "gausslegendre"

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.rmax <= 0 or self.rstep <= 0:
            raise ValueError("rmax and rstep must be positive")
        if self.smin >= self.smax:
            raise ValueError("smin must be less than smax")
        if self.quad_points < 2:
            raise ValueError("quad_points must be at least 2")
        if self.integration_method not in ["gausslegendre", "simpson", "trapezoid"]:
            raise ValueError("integration_method must be 'gausslegendre', 'simpson' or 'trapezoid'")


class XMap_torch:
    """Torch version of qFit XMap for handling crystallographic symmetry."""

    def __init__(
        self,
        xmap: XMap = None,
        array: torch.Tensor | None = None,
        grid_parameters: GridParameters | None = None,
        unit_cell: UnitCell | None = None,
        resolution: Resolution | float | None = None,
        hkl: torch.Tensor | None = None,
        origin: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize differentiable XMap.

        Parameters
        ----------
        xmap: XMap, optional
            Initialize from qFit XMap (preferred).
        array: Optional[torch.Tensor], optional
            Map array, by default None.
        grid_parameters: Optional[GridParameters], optional
            Grid parameters for the map, meaning voxel spacing (voxelspacing) and
            offset.
        unit_cell: Optional[UnitCell], optional
            Crystallographic unit cell information.
        resolution: Optional[Union[Resolution, float]], optional
            Map resolution in Angstroms, by default None.
        hkl: Optional[torch.Tensor], optional
            Miller indices for the map, by default None.
        device: torch.device, optional
            Device to use for computations, by default 'cpu'.
        """
        if xmap is not None:
            self.unit_cell = xmap.unit_cell
            self.resolution = xmap.resolution
            self.hkl = xmap.hkl
            self.origin = xmap.origin
            self.array = torch.tensor(xmap.array, dtype=torch.float32, device=device)
            self.shape = xmap.shape

            self.voxelspacing = torch.tensor(xmap.voxelspacing, device=device)
            self.offset = torch.tensor(xmap.offset, device=device)
            self.xmap = xmap
        else:
            self.unit_cell = unit_cell
            self.resolution = resolution
            self.hkl = hkl
            self.origin = origin if origin is not None else torch.zeros(3, device=device)
            self.array = array.float()
            self.shape = self.array.shape

            self.voxelspacing = torch.tensor(grid_parameters.voxelspacing, device=device)
            self.offset = torch.tensor(grid_parameters.offset, device=device)

            self.xmap = XMap(array, grid_parameters, unit_cell, resolution, hkl, origin)

        self._validate_input(xmap)

        self._setup_symmetry_matrices(device)

    def _validate_input(self, xmap) -> None:
        """Validate input parameters."""
        if xmap is not None:
            if not isinstance(xmap, XMap):
                raise ValueError("xmap must be an instance of qFit's XMap (adp3d.qfit.volume.XMap)")
            if not hasattr(self.resolution, "high"):
                warnings.warn(
                    "resolution is not a Resolution object (is "
                    f"{type(self.resolution)}), using provided resolution as high and "
                    "1000 Å as low"
                )
                self.resolution = Resolution(high=self.resolution, low=1000.0)
            if self.resolution.low is None:
                warnings.warn("resolution does not have low limit set, using 1000 Å as low")
                self.resolution.low = 1000.0
        else:
            if self.array is None:
                raise ValueError("array must be provided")
            if self.unit_cell is None:
                raise ValueError("unit_cell must be provided")
            if self.resolution is None:
                raise ValueError("resolution must be provided")
            if self.voxelspacing is None:
                raise ValueError("grid_parameters must be provided")
            if self.offset is None:
                raise ValueError("grid_parameters must be provided")
            if not hasattr(self.resolution, "high"):
                warnings.warn(
                    "resolution is not a Resolution object "
                    f"(is {type(self.resolution)}), using provided resolution as high "
                    "and 1000 Å as low"
                )
                self.resolution = Resolution(high=self.resolution, low=1000.0)
            if not hasattr(self.resolution, "low") and isinstance(self.resolution, Resolution):
                warnings.warn("resolution does not have a low attribute, using 1000 Å as low")
                self.resolution.low = 1000.0

            if (
                torch.all(self.array == torch.zeros(self.array.shape, device=self.array.device))
                and self.hkl is None
            ):
                warnings.warn(
                    f"hkl ({self.hkl}) is not provided and array is zeros. \
                    If this is intended to contain structure factors for computing a "
                    "map, please provide hkl."
                )

    def _setup_symmetry_matrices(self, device: torch.device) -> None:
        """Precompute symmetry operation matrices for efficient application."""
        symops = self.unit_cell.space_group.symop_list
        n_ops = len(symops)

        R_matrices = torch.zeros((n_ops, 3, 3), device=device)
        t_vectors = torch.zeros((n_ops, 3), device=device)

        for i, symop in enumerate(symops):
            R_matrices[i] = torch.tensor(symop.R, device=device, dtype=torch.float32)
            t_vectors[i] = torch.tensor(symop.t, device=device, dtype=torch.float32)

        self.R_matrices = R_matrices
        self.t_vectors = t_vectors

    def tofile(self, filename: str, density: torch.Tensor | NDArray = None) -> None:
        """Save the map to a file.

        Parameters
        ----------
        filename: str
            Output filename.
        density: Union[torch.Tensor, NDArray], optional
            Density grid to save, by default None.
            If provided, it will be used to update the map array.
        """
        if density is not None:
            if density.shape != self.array.shape:
                raise ValueError(
                    f"Density shape {density.shape} does not match map shape {self.array.shape}"
                )
            if isinstance(density, torch.Tensor):
                density = density.cpu().numpy()
        else:
            density = self.array.cpu().numpy()

        xmap_writer = deepcopy(self.xmap)
        xmap_writer.array = density
        xmap_writer.tofile(filename)

    def downsample_to_resolution(
        self,
        target_resolution: float,
        apply_filter: bool = True,
        filter_type: str = "brickwall",
    ) -> "XMap_torch":
        """Resample density map to a target resolution using tricubic interpolation.

        Parameters
        ----------
        target_resolution: float
            Target resolution in Angstroms.
        apply_filter: bool, optional
            Apply Gaussian filter when downsampling to prevent aliasing, by default
            True.
        filter_type: str, optional
            Type of filter to apply ('hamming' or 'brickwall'), by default 'brickwall'.

        Returns
        -------
        XMap_torch
            New XMap_torch instance with resampled density map.
        """
        current_resolution = self.resolution.high
        unit_cell = self.unit_cell
        device = self.array.device

        new_grid_a = int(np.ceil(unit_cell.a / target_resolution * 4.0))
        new_grid_b = int(np.ceil(unit_cell.b / target_resolution * 4.0))
        new_grid_c = int(np.ceil(unit_cell.c / target_resolution * 4.0))
        new_shape = torch.tensor([new_grid_a, new_grid_b, new_grid_c], device=device)

        scale_factor = target_resolution / current_resolution

        new_voxelspacing = torch.tensor([target_resolution / 4.0] * 3, device=device)
        new_offset = (self.offset / scale_factor).detach().clone()

        if scale_factor < 1:
            warnings.warn(
                f"Target resolution ({target_resolution} Å) higher than current "
                f"resolution ({current_resolution} Å). Upsampling will not add "
                "information."
            )

        if scale_factor > 1 and apply_filter:
            orig_shape = self.array.shape
            f_density = to_f_density(self.array)

            shape = torch.tensor(f_density.shape, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(
                [torch.arange(-(s // 2), -(s // 2) + s, device=device) for s in shape],
                indexing="ij",
            )

            # Calculate physical frequencies based on voxel spacing
            freq_x = grid_x / (shape[0] * self.voxelspacing[0])
            freq_y = grid_y / (shape[1] * self.voxelspacing[1])
            freq_z = grid_z / (shape[2] * self.voxelspacing[2])

            radial_freq = torch.sqrt(freq_x**2 + freq_y**2 + freq_z**2)
            cutoff_freq = 1.0 / target_resolution

            if filter_type == "hamming":
                resolution_filter = radial_hamming_3d(radial_freq, cutoff_freq)
            elif filter_type == "brickwall":
                resolution_filter = (radial_freq < cutoff_freq).float()

            f_density = f_density * resolution_filter
            # crop in case padding occurred in to_f_density
            array_for_sampling = to_density(f_density)[
                ..., : orig_shape[0], : orig_shape[1], : orig_shape[2]
            ]
        else:
            array_for_sampling = self.array

        original_shape = torch.tensor(self.shape, device=device, dtype=torch.float32)

        x_norm = (torch.arange(new_shape[0], device=device, dtype=torch.float32)) / new_shape[
            0
        ].float()
        y_norm = (torch.arange(new_shape[1], device=device, dtype=torch.float32)) / new_shape[
            1
        ].float()
        z_norm = (torch.arange(new_shape[2], device=device, dtype=torch.float32)) / new_shape[
            2
        ].float()

        z_orig = z_norm * original_shape[0]
        y_orig = y_norm * original_shape[1]
        x_orig = x_norm * original_shape[2]

        grid_z, grid_y, grid_x = torch.meshgrid(z_orig, y_orig, x_orig, indexing="ij")
        points_zyx = torch.stack([grid_z, grid_y, grid_x], dim=-1)

        resampled_array = tricubic_interpolation_torch(array_for_sampling, points_zyx)

        new_grid_parameters = GridParameters(
            voxelspacing=new_voxelspacing.cpu().numpy(),
            offset=new_offset.cpu().numpy(),
        )

        new_resolution = Resolution(high=target_resolution, low=self.resolution.low)

        return XMap_torch(
            array=resampled_array,
            grid_parameters=new_grid_parameters,
            unit_cell=self.unit_cell,
            resolution=new_resolution,
            hkl=self.hkl,
            origin=self.origin,
            device=self.array.device,
        )


class DifferentiableTransformer(torch.nn.Module):
    """Differentiable transformation of atomic coordinates to electron density.

    Implements a fully differentiable pipeline for converting atomic coordinates
    to electron density maps with crystallographic symmetry operations. Supports
    both X-ray and electron microscopy modes with flexible parameter configuration.
    """

    def __init__(
        self,
        xmap: XMap_torch,
        scattering_params: torch.Tensor,
        density_params: DensityParameters | None = None,
        em: bool = False,
        space_group: int | None = None,
        device: torch.device = torch.device("cpu"),
        use_cuda_kernels: bool = True,
    ) -> None:
        """Initialize differentiable transformer.

        Parameters
        ----------
        xmap: DifferentiableXMap
            Differentiable XMap object.
        scattering_params: torch.Tensor
            Atomic scattering parameters for each element, of shape [n_elem, n_coeffs,
            2]. NOTE: The indexing on the elements MUST match the element indices in the
            input to forward!
        density_params: Optional[DensityParameters], optional
            Parameters for density calculation, by default None.
            Min and max scattering vector magnitudes will be updated by resolutions in
            the XMap.
        em: bool, optional
            Whether to use electron microscopy mode, by default False.
        space_group: Optional[int], optional
            Space group number, by default None (in which case it is expected from the
            XMap unit cell).
        device: torch.device, optional
            Device to use for computations, by default 'cpu'.
        use_cuda_kernels: bool, optional
            Whether to use CUDA kernels for performance, by default False.
        """
        super().__init__()
        self.device = device
        self.xmap = xmap
        self.unit_cell = xmap.unit_cell
        if space_group is not None:
            self.unit_cell.space_group = GetSpaceGroup(space_group)

        self.grid_shape = xmap.shape

        # TODO: currently this means we're forced to use the existing B-factors
        self.scattering_params = scattering_params.to(device)
        self.density_params = density_params or DensityParameters()
        self.em = em

        self.xmap = xmap

        self.density_params.smax = 1 / (2 * self.xmap.resolution.high)
        self.density_params.smin = 1 / (2 * self.xmap.resolution.low)

        self.integrator = GaussLegendreQuadrature(
            num_points=self.density_params.quad_points,
            device=self.device,
            dtype=self.scattering_params.dtype,
        )
        self.use_cuda_kernels = use_cuda_kernels
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        # need to set this here or else doubles start popping up and ruining operations
        self.dtype = torch.float32

        lattice_to_cartesian = self.xmap.unit_cell.frac_to_orth / self.xmap.unit_cell.abc
        cartesian_to_lattice = self.xmap.unit_cell.orth_to_frac * self.xmap.unit_cell.abc.reshape(
            3, 1
        )
        grid_to_cartesian = lattice_to_cartesian * self.xmap.voxelspacing.cpu().numpy()
        self.register_buffer(
            "lattice_to_cartesian",
            torch.tensor(lattice_to_cartesian).to(dtype=self.dtype, device=self.device),
        )

        self.register_buffer(
            "cartesian_to_lattice",
            torch.tensor(cartesian_to_lattice).to(dtype=self.dtype, device=self.device),
        )

        self.register_buffer(
            "grid_to_cartesian",
            torch.tensor(grid_to_cartesian).to(dtype=self.dtype, device=self.device),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        active: torch.Tensor | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing electron density with symmetry.

        Parameters
        ----------
        coordinates: torch.Tensor
            Atomic coordinates of shape (batch_size, n_atoms, 3).
        elements: torch.Tensor
            Element indices of shape (batch_size, n_atoms).
            NOTE: The indexing on the elements MUST match the element indices in the
            input to forward!
        b_factors: torch.Tensor
            B-factors of shape (batch_size, n_atoms).
        occupancies: torch.Tensor
            Occupancies of shape (batch_size, n_atoms).
        active: Optional[torch.Tensor], optional
            Boolean mask of active atoms of shape (batch_size, n_atoms). Defaults to
            all True if None.
        unique_combinations: torch.Tensor | None, optional
            Pre-computed unique (element, b_factor) pairs for vmap compatibility
        inverse_indices: torch.Tensor | None, optional
            Pre-computed inverse indices for vmap compatibility

        Returns
        -------
        torch.Tensor
            Symmetry-expanded density grid of shape (batch_size, *grid_shape).
        """
        if not (
            coordinates.shape[0]
            == elements.shape[0]
            == b_factors.shape[0]
            == occupancies.shape[0]
            == (active.shape[0] if active is not None else coordinates.shape[0])
        ):
            raise ValueError("Batch sizes must match for all inputs")

        coordinates = coordinates.to(self.device)

        # Needs batch dimension for einsum: ~line 579
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        elements = elements.to(self.device)
        b_factors = b_factors.to(self.device)
        occupancies = occupancies.to(self.device)
        active = (
            torch.ones_like(elements, dtype=torch.bool, device=self.device)
            if active is None
            else active.to(dtype=torch.bool, device=self.device)
        )

        grid_coordinates = self._compute_grid_coordinates(coordinates).to(dtype=torch.float32)

        lmax = torch.tensor(
            [self.density_params.rmax / vs for vs in self.xmap.voxelspacing],
            device=self.device,
        )

        if self.use_cuda_kernels:
            radial_densities, radial_derivatives = self._compute_radial_derivatives(
                elements, b_factors, unique_combinations, inverse_indices
            )
            radial_densities = radial_densities.to(dtype=self.dtype).float()
            radial_derivatives = radial_derivatives.to(dtype=self.dtype).float()

            final_density = dilate_atom_centric(
                self._symmetry_expand(grid_coordinates),
                occupancies,
                radial_densities,
                radial_derivatives,
                self.density_params.rstep,
                self.density_params.rmax,
                lmax,
                self.grid_shape,
                self.grid_to_cartesian,
            )
        else:
            radial_densities = self._compute_radial_densities(
                elements, b_factors, unique_combinations, inverse_indices
            ).to(dtype=self.dtype)
            # Dilate every symmetry copy of every atom, as the CUDA kernel does: flatten
            # [batch, ops, atoms, 3] to [batch, ops * atoms, 3] (ops outer, atoms inner)
            # and repeat the per-atom inputs over ops in the same order.
            n_ops = len(self.xmap.R_matrices)
            symmetry_coordinates = self._symmetry_expand(grid_coordinates)
            final_density = dilate_points_torch(
                symmetry_coordinates.reshape(symmetry_coordinates.shape[0], -1, 3),
                active.repeat(1, n_ops),
                occupancies.repeat(1, n_ops),
                lmax,
                radial_densities.repeat(1, n_ops, 1),
                self.density_params.rstep,
                self.density_params.rmax,
                self.grid_to_cartesian,
                self.grid_shape,
            )

        return final_density

    def _symmetry_expand(self, grid_coordinates: torch.Tensor) -> torch.Tensor:
        """Apply the space group's symmetry operations to atom grid coordinates.

        Parameters
        ----------
        grid_coordinates: torch.Tensor
            Grid coordinates in (a, b, c) order, of shape (batch_size, n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Grid coordinates of every symmetry copy, in (a, b, c) order, of shape
            (batch_size, n_ops, n_atoms, 3).
        """
        grid_shape_tensor = torch.tensor(self.grid_shape, device=self.device)

        grid_coor_rot = torch.einsum(
            "rji,bni->brnj",
            self.xmap.R_matrices,
            grid_coordinates,
        )

        # Apply fractional translation scaled by (nx, ny, nz)
        grid_coor_rot += self.xmap.t_vectors.unsqueeze(1) * grid_shape_tensor.flip(0).view(1, 1, 3)
        return grid_coor_rot

    def create_mask(self, coordinates: torch.Tensor, radius: float) -> torch.Tensor:
        """Create a boolean mask volume around atoms within a radius, considering
        symmetry.

        Symmetry copies of the atoms are generated the same way as in forward(), and
        voxels within the radius of any copy are marked, wrapping periodically at the
        grid boundaries.

        Parameters
        ----------
        coordinates: torch.Tensor
            Cartesian coordinates of atoms, shape (n_atoms, 3).
            Assumes input coordinates are on the correct device.
        radius: float
            Radius in Ångstroms to carve around atoms.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape (grid_shape).
        """
        device = self.device
        dtype = self.dtype
        grid_shape = self.grid_shape  # Shape: tuple (nz, ny, nx)
        n_atoms = coordinates.shape[0]  # Shape: [n_atoms]
        radius2 = radius * radius

        if n_atoms == 0:
            return torch.zeros(grid_shape, device=device, dtype=torch.bool)

        coordinates = coordinates.to(device=device, dtype=dtype)

        # Carve around every symmetry copy of every atom, as forward() dilates them.
        grid_coordinates = self._symmetry_expand(
            self._compute_grid_coordinates(coordinates).reshape(1, -1, 3)
        ).reshape(-1, 3)  # Shape: [n_ops * n_atoms, 3] (a, b, c order)

        min_voxel_spacing = torch.min(self.xmap.voxelspacing.to(device))
        max_extent_voxel = torch.ceil((radius + 1e-6) / min_voxel_spacing).int().item()

        nearby_grid_indices = torch.arange(
            -max_extent_voxel, max_extent_voxel + 1, device=device, dtype=torch.int
        )
        nearby_grid_offsets = torch.stack(
            torch.meshgrid(
                nearby_grid_indices,
                nearby_grid_indices,
                nearby_grid_indices,
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)  # Shape: [n_nearby, 3]

        mask_volume = torch.zeros(grid_shape, device=device, dtype=torch.bool)
        grid_shape_abc = torch.tensor(grid_shape[::-1], device=device, dtype=torch.int)

        coord_floored = torch.floor(grid_coordinates).int()  # Shape: [n_atoms, 3]
        grid_points_absolute = coord_floored.unsqueeze(1) + nearby_grid_offsets.unsqueeze(
            0
        )  # Shape: [n_atoms, n_nearby, 3]

        delta_grid = grid_coordinates.unsqueeze(1) - grid_points_absolute.to(
            dtype
        )  # Shape: [n_atoms, n_nearby, 3]

        delta_cartesian = delta_grid @ self.grid_to_cartesian.T  # Shape: [n_atoms, n_nearby, 3]

        distances2 = torch.sum(
            delta_cartesian * delta_cartesian, dim=-1
        )  # Shape: [n_atoms, n_nearby]

        within_radius_mask = distances2 <= radius2  # Shape: [n_atoms, n_nearby]

        points_to_mark_absolute = grid_points_absolute[
            within_radius_mask
        ]  # Shape: [n_valid_points, 3]

        points_to_mark_wrapped = (
            points_to_mark_absolute % grid_shape_abc
        )  # Shape: [n_valid_points, 3] (a, b, c order)
        mask_volume[
            points_to_mark_wrapped[:, 2], points_to_mark_wrapped[:, 1], points_to_mark_wrapped[:, 0]
        ] = True

        return mask_volume

    def _compute_radial_densities(
        self,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute radial densities using numerical integration.

        Parameters
        ----------
        elements: torch.Tensor
            Element indices of shape (batch_size, n_atoms).
        b_factors: torch.Tensor
            B-factors of shape (batch_size, n_atoms).
        unique_combinations: torch.Tensor | None, optional
            Pre-computed unique (element, b_factor) pairs for vmap compatibility
        inverse_indices: torch.Tensor | None, optional
            Pre-computed inverse indices for vmap compatibility

        Returns
        -------
        torch.Tensor
            Radial densities of shape (batch_size, n_atoms, n_radial).
        """
        r = torch.arange(
            0,
            self.density_params.rmax + self.density_params.rstep,
            self.density_params.rstep,
            device=self.device,
        )

        batch_size, n_atoms = elements.shape if elements.ndim == 2 else (1, elements.shape[0])
        n_radial = r.shape[0]

        if unique_combinations is None or inverse_indices is None:
            elements_flat = elements.reshape(-1)
            b_factors_flat = b_factors.reshape(-1)
            combined = torch.stack([elements_flat, b_factors_flat], dim=1)
            unique_combinations, inverse_indices = torch.unique(
                combined, dim=0, return_inverse=True
            )
        else:
            unique_combinations = unique_combinations.to(self.device)
            inverse_indices = inverse_indices.to(self.device)

        unique_elements = unique_combinations[:, 0].int()
        element_asf = self.scattering_params[unique_elements]  # Shape: [n_unique, n_coeffs, 2]
        unique_bfactors = unique_combinations[:, 1]

        def integrate_single_element(asf: torch.Tensor, bfac: torch.Tensor) -> torch.Tensor:
            """Compute density for a batch of parameters and b-factors."""

            def integrand_fn(s):
                return scattering_integrand(
                    s,
                    r,
                    asf,
                    bfac,
                    em=self.em,
                )

            result = self.integrator(
                integrand_fn,
                integration_limits=torch.tensor(
                    [[self.density_params.smin, self.density_params.smax]],
                    device=self.device,
                ),
                dim=1,
            )

            return result  # Shape: [batch, n_radial]

        # Create a vmap that processes all unique combinations at once
        integrate_elements = torch.vmap(integrate_single_element)
        all_unique_densities = integrate_elements(element_asf, unique_bfactors)

        densities = all_unique_densities[inverse_indices].reshape(batch_size, n_atoms, n_radial)

        return densities

    def _compute_radial_derivatives(
        self,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute radial density derivatives efficiently.

        Parameters
        ----------
        elements: torch.Tensor
            Element indices of shape (batch_size, n_atoms).
        b_factors: torch.Tensor
            B-factors of shape (batch_size, n_atoms).
        unique_combinations: torch.Tensor | None, optional
            Pre-computed unique (element, b_factor) pairs for vmap compatibility
        inverse_indices: torch.Tensor | None, optional
            Pre-computed inverse indices for vmap compatibility

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Radial densities and their approximate derivatives of shape
            (batch_size, n_atoms, n_radial).
        """
        densities = self._compute_radial_densities(
            elements, b_factors, unique_combinations, inverse_indices
        )

        r = torch.arange(
            0,
            self.density_params.rmax + self.density_params.rstep,
            self.density_params.rstep,
            device=self.device,
        )

        batch_size, n_atoms = elements.shape if elements.ndim == 2 else (1, elements.shape[0])

        if unique_combinations is None or inverse_indices is None:
            elements_flat = elements.reshape(-1)
            b_factors_flat = b_factors.reshape(-1)
            combined = torch.stack([elements_flat, b_factors_flat], dim=1)
            unique_combinations, inverse_indices = torch.unique(
                combined, dim=0, return_inverse=True
            )
        else:
            unique_combinations = unique_combinations.to(self.device)
            inverse_indices = inverse_indices.to(self.device)

        unique_elements = unique_combinations[:, 0].int()
        element_asf = self.scattering_params[unique_elements]
        unique_bfactors = unique_combinations[:, 1]

        def integrate_single_element_derivative(
            asf: torch.Tensor, bfac: torch.Tensor
        ) -> torch.Tensor:
            """Compute density derivative for a single element and b-factor."""

            def integrand_fn(s):
                return scattering_integrand_derivative(
                    s,
                    r,
                    asf,
                    bfac,
                    em=self.em,
                )

            result = self.integrator(
                integrand_fn,
                integration_limits=torch.tensor(
                    [[self.density_params.smin, self.density_params.smax]],
                    device=self.device,
                ),
                dim=1,
            )
            return result

        integrate_elements = torch.vmap(integrate_single_element_derivative)
        all_unique_derivatives = integrate_elements(element_asf, unique_bfactors)

        derivatives = all_unique_derivatives[inverse_indices].reshape(
            batch_size, n_atoms, r.shape[0]
        )

        return densities, derivatives

    def _compute_grid_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Transform Cartesian coordinates to grid coordinates.

        Parameters
        ----------
        coordinates: torch.Tensor
            Cartesian coordinates of shape (batch_size, n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Grid coordinates of shape (batch_size, n_atoms, 3).
        """
        if hasattr(self.xmap, "origin") and not torch.allclose(
            torch.tensor(self.xmap.origin, device=self.device, dtype=coordinates.dtype),
            torch.zeros(3, device=self.device),
        ):
            coordinates = coordinates - torch.tensor(
                self.xmap.origin, device=self.device, dtype=coordinates.dtype
            )

        grid_coordinates = torch.matmul(coordinates, self.cartesian_to_lattice.T)
        grid_coordinates /= self.xmap.voxelspacing.to(self.device)

        if hasattr(self.xmap, "offset"):
            grid_coordinates -= self.xmap.offset.to(device=self.device, dtype=coordinates.dtype)

        return grid_coordinates


def dilate_points_torch(
    coordinates: torch.Tensor,
    active: torch.Tensor,
    occupancies: torch.Tensor,
    lmax: torch.Tensor,
    radial_densities: torch.Tensor,
    rstep: float,
    rmax: float,
    grid_to_cartesian: torch.Tensor,
    grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Point dilation onto a grid across batches.
    Batch could be either many structures for many density maps,
    or an ensemble, where you would need to sum across the output batch to get a map.

    Parameters
    ----------
    coordinates: torch.Tensor
        Grid coordinates corresponding to atom positions of shape
        (batch_size, n_atoms, 3).
    active: torch.Tensor
        Boolean mask of active atoms of shape (batch_size, n_atoms).
    occupancies: torch.Tensor
        Occupancies of shape (batch_size, n_atoms).
    lmax: torch.Tensor
        Maximum distances from atom to consider in grid units of shape (3,).
    radial_densities: torch.Tensor
        Precomputed radial densities of shape (batch_size, n_atoms, n_radial).
    rstep: float
        Step size for radial grid.
    rmax: float
        Maximum radius for density calculation.
    grid_to_cartesian: torch.Tensor
        Transformation matrix from grid to Cartesian of shape (3, 3).
    grid_shape: tuple[int, int, int]
        Output grid shape (int, int, int).

    Returns
    -------
    torch.Tensor
        Density grid (batch_size, *grid_shape).
    """
    device = coordinates.device
    dtype = coordinates.dtype
    batch_size, n_atoms = coordinates.shape[:2]

    result = torch.zeros((batch_size,) + grid_shape, device=device, dtype=dtype)

    max_extents = [int(torch.ceil(lmax[i]).item()) for i in range(3)]
    nearby_grid = torch.stack(
        torch.meshgrid(
            [  # c, b, a ordered
                torch.arange(-max_extents[2], max_extents[2] + 1, device=device),
                torch.arange(-max_extents[1], max_extents[1] + 1, device=device),
                torch.arange(-max_extents[0], max_extents[0] + 1, device=device),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)  # [n_nearby, 3]

    # Compute the a, b, c distance from the grid voxels for each atom
    grid_difference = coordinates - torch.floor(coordinates)  # [batch_size, n_atoms, 3]

    # Transform grid to a, b, c for subtraction
    nearby_grid_abc = nearby_grid[:, [2, 1, 0]]  # [n_nearby, 3]

    # Compute the distance from the grid for each atom to each nearby grid point
    # in the offset
    delta_to_nearby = grid_difference.view(batch_size, 1, n_atoms, 3) - nearby_grid_abc.view(
        1, -1, 1, 3
    )  # [batch_size, n_nearby, n_atoms, 3]
    cartesian_delta_to_nearby = torch.matmul(
        delta_to_nearby, grid_to_cartesian.T
    )  # [batch_size, n_nearby, n_atoms, 3]
    distances_to_nearby = torch.linalg.norm(
        cartesian_delta_to_nearby, dim=-1
    )  # [batch_size, n_nearby, n_atoms]

    rad_continuous = distances_to_nearby / rstep  # [batch_size, n_nearby, n_atoms]
    rad_indices_low = torch.floor(rad_continuous).int()  # [batch_size, n_nearby, n_atoms]
    weights_high = rad_continuous - rad_indices_low.float()  # [batch_size, n_nearby, n_atoms]
    weights_low = 1.0 - weights_high  # [batch_size, n_nearby, n_atoms]

    # Clamp to valid range
    max_rad_idx = radial_densities.shape[-1] - 1
    rad_indices_low = torch.clamp(
        rad_indices_low, 0, max_rad_idx
    )  # [batch_size, n_nearby, n_atoms]
    rad_indices_high = torch.clamp(
        rad_indices_low + 1, 0, max_rad_idx
    )  # [batch_size, n_nearby, n_atoms]

    active_mask = (
        active.reshape(batch_size * n_atoms).nonzero().squeeze(-1)
    )  # [n_nearby * n_atoms].nonzero() -> [n_active_atoms]
    batch_idx = active_mask // n_atoms  # [n_active_atoms]
    atom_idx = active_mask % n_atoms  # [n_active_atoms]

    # Calculate grid points to interpolate onto
    n_active_atoms = len(active_mask)
    n_nearby = nearby_grid.shape[0]
    coord_floored = torch.floor(coordinates[batch_idx, atom_idx]).int()  # [n_active_atoms, 3]

    # modulo for periodic boundary
    grid_points = (coord_floored.view(-1, 1, 3) + nearby_grid_abc.view(1, -1, 3)) % torch.tensor(
        grid_shape[::-1], device=device
    )  # [n_active_atoms, n_nearby, 3]

    atom_indices = torch.arange(n_active_atoms, device=device).repeat_interleave(n_nearby)
    offset_indices = torch.arange(n_nearby, device=device).repeat(n_active_atoms)
    final_batch_indices = batch_idx[atom_indices]
    final_atom_indices = atom_idx[atom_indices]  # all [n_active_atoms * n_nearby]

    rad_indices_low_final = rad_indices_low[final_batch_indices, offset_indices, final_atom_indices]
    rad_indices_high_final = rad_indices_high[
        final_batch_indices, offset_indices, final_atom_indices
    ]

    within_rmax = distances_to_nearby <= rmax  # [batch_size, n_nearby, n_atoms]

    # Interpolate radial densities onto grid and scale by occupancy; the search box reaches
    # past rmax at its corners, so zero those points as the CUDA kernel does.
    densities = (
        weights_low[final_batch_indices, offset_indices, final_atom_indices]
        * radial_densities[final_batch_indices, final_atom_indices, rad_indices_low_final]
        + weights_high[final_batch_indices, offset_indices, final_atom_indices]
        * radial_densities[final_batch_indices, final_atom_indices, rad_indices_high_final]
    ) * (
        occupancies[final_batch_indices, final_atom_indices]
        * within_rmax[final_batch_indices, offset_indices, final_atom_indices]
    )  # [n_active_atoms * n_nearby]

    # scatter_add_ onto the grid
    grid_points_flat = grid_points.reshape(-1, 3).int()  # [n_active_atoms * n_nearby, 3]
    grid_strides = [
        grid_shape[1] * grid_shape[2],
        grid_shape[2],
        1,
    ]  # [grid_y * grid_x, grid_x, 1]

    # Calculate strided indices for proper scatter_add_
    flat_grid_indices = (
        final_batch_indices * (grid_shape[0] * grid_shape[1] * grid_shape[2])  # batch
        + grid_points_flat[:, 2] * grid_strides[0]  # z
        + grid_points_flat[:, 1] * grid_strides[1]  # y
        + grid_points_flat[:, 0]  # x
    ).long()  # [n_active_atoms * n_nearby]

    result.view(-1).scatter_add_(
        0, flat_grid_indices, densities
    )  # Add onto [batch_size * grid_z * grid_y * grid_x] shape

    return result


def scattering_integrand(
    s: torch.Tensor,
    r: torch.Tensor,
    asf: torch.Tensor,
    bfactor: torch.Tensor,
    em: bool = False,
) -> torch.Tensor:
    """Compute the scattering integrand for radial density calculation.

    Parameters
    ----------
    s: torch.Tensor
        Scattering vector magnitudes, shape (..., n_s).
    r: torch.Tensor
        Radial distances, shape (n_r,).
    asf: torch.Tensor
        Atomic scattering factors with shape (..., n_coeffs, 2).
        The coefficients are [a_i, b_i] pairs for the scattering model.
    bfactor: torch.Tensor
        B-factors, shape (...,).
    em: bool, optional
        Whether to use electron microscopy mode, by default False.

    Returns
    -------
    torch.Tensor
        Computed integrand values with shape (..., n_s, n_r).
    """
    s_expanded = s.reshape(*s.shape, 1)  # [..., n_s, 1]
    r_expanded = r.reshape(1, -1)  # [1, n_r]

    s2 = s_expanded * s_expanded  # [..., n_s, 1]

    bfactor_expanded = bfactor.reshape(*bfactor.shape, 1, 1)  # [..., 1, 1]

    if em:
        # electron microscopy mode (no constant term)
        a_coeffs = asf[..., :5, 0]  # [..., 5]
        b_coeffs = asf[..., :5, 1]  # [..., 5]

        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)  # [..., 5, 1, 1]
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)  # [..., 5, 1, 1]

        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))  # [..., 5, n_s, 1]

        f = torch.sum(a_coeffs * exp_terms, dim=-3)  # [..., n_s, 1]
    else:
        # X-ray scattering mode (includes constant term)
        a_coeffs = asf[..., :5, 0]  # [..., 5]
        b_coeffs = asf[..., :5, 1]  # [..., 5]
        constant_term = asf[..., 5, 0].reshape(*asf.shape[:-2], 1, 1)  # [..., 1, 1]

        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)  # [..., 5, 1, 1]
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)  # [..., 5, 1, 1]

        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))  # [..., 5, n_s, 1]

        f = torch.sum(a_coeffs * exp_terms, dim=-3) + constant_term  # [..., n_s, 1]

    four_pi_s = 4 * torch.pi * s_expanded  # [..., n_s, 1]
    w = 8 * f * torch.exp(-bfactor_expanded * s2) * s_expanded  # [..., n_s, 1]

    eps = 1e-4
    r_small_mask = (r_expanded < eps).expand(
        s_expanded.shape[:-1] + r_expanded.shape[-1:]
    )  # [..., n_s, n_r]

    ar = four_pi_s * r_expanded  # [..., n_s, n_r]
    ar2 = ar * ar

    # prevent singularity with 4th order Taylor expansion
    taylor_term = 1.0 - ar2 / 6.0
    small_r_values = w * four_pi_s * taylor_term  # [..., n_s, n_r]

    sin_term = torch.sin(ar)

    safe_r = torch.where(r_expanded > 0, r_expanded, torch.ones_like(r_expanded))
    large_r_values = w * sin_term / safe_r  # [..., n_s, n_r]

    result = torch.where(r_small_mask, small_r_values, large_r_values)

    return result


def scattering_integrand_derivative(
    s: torch.Tensor,
    r: torch.Tensor,
    asf: torch.Tensor,
    bfactor: torch.Tensor,
    em: bool = False,
) -> torch.Tensor:
    """Compute the derivative of scattering integrand with respect to radius.

    Parameters are the same as ``scattering_integrand``, but this calculates
    the analytical derivative with respect to ``r``.
    """
    s_expanded = s.reshape(*s.shape, 1)
    r_expanded = r.reshape(1, -1)
    s2 = s_expanded * s_expanded
    bfactor_expanded = bfactor.reshape(*bfactor.shape, 1, 1)

    if em:
        a_coeffs = asf[..., :5, 0]
        b_coeffs = asf[..., :5, 1]
        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)
        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))
        f = torch.sum(a_coeffs * exp_terms, dim=-3)
    else:
        a_coeffs = asf[..., :5, 0]
        b_coeffs = asf[..., :5, 1]
        constant_term = asf[..., 5, 0].reshape(*asf.shape[:-2], 1, 1)
        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)
        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))
        f = torch.sum(a_coeffs * exp_terms, dim=-3) + constant_term

    four_pi_s = 4 * torch.pi * s_expanded
    w = 8 * f * torch.exp(-bfactor_expanded * s2) * s_expanded

    eps = 1e-4
    r_small_mask = (r_expanded < eps).expand(s_expanded.shape[:-1] + r_expanded.shape[-1:])

    ar = four_pi_s * r_expanded
    ar2 = ar * ar

    # For derivatives, calculate cos and sin terms
    cos_term = torch.cos(ar)
    sin_term = torch.sin(ar)

    # For r > eps: w/r * (a*cos(ar) - sin(ar)/r)
    safe_r = torch.where(r_expanded > 0, r_expanded, torch.ones_like(r_expanded))
    large_r_values = w * (four_pi_s * cos_term - sin_term / safe_r) / safe_r

    # prevent singularity with Taylor expansion
    a3 = four_pi_s * four_pi_s * four_pi_s
    small_r_values = w * a3 * r_expanded * (ar2 - 8) / 24.0

    result = torch.where(r_small_mask, small_r_values, large_r_values)
    return result


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to a Gaussian with mean 0 and std dev 1."""
    if t.numel() == 0 or t.ndim == 0:
        return t

    if t.dtype not in [torch.complex32, torch.complex64, torch.complex128]:
        return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)

    real_part = (t.real - t.real.mean()) / (t.real.std(unbiased=False) + 1e-8)
    imag_part = (t.imag - t.imag.mean()) / (t.imag.std(unbiased=False) + 1e-8)

    return torch.view_as_complex(torch.cat([real_part[..., None], imag_part[..., None]], -1))


def scale_map(
    xmap_array: torch.Tensor,
    model_map_array: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scale xmap_array to match the mean and standard deviation of model_map_array
    within the mask.

    The scale is always positive and does not depend on the units of xmap_array, so
    the target keeps the map's contrast however well the model fits it. Unlike a
    least-squares fit, it does not shrink the map by its correlation with the model.

    Parameters
    ----------
    xmap_array: torch.Tensor
        The experimental map array to be scaled.
    model_map_array: torch.Tensor
        The calculated model map array used as the reference for scaling.
    mask: torch.Tensor
        A boolean mask indicating the regions to use for calculating scaling factors.

    Returns
    -------
    torch.Tensor
        The scaled xmap_array.

    Raises
    ------
    ValueError
        If xmap_array or model_map_array is constant within the mask, or the mask
        selects a single voxel.
    """
    if not mask.any():
        warnings.warn("Mask for map scaling is empty. Using all map.")
        mask = torch.ones_like(xmap_array).bool()

    xmap_masked = xmap_array[mask]
    model_masked = model_map_array[mask]

    xmap_std, model_std = xmap_masked.std(), model_masked.std()
    # NaN for a one-voxel mask, so test for a positive spread rather than a nonzero one.
    if not (xmap_std > 0 and model_std > 0):
        raise ValueError("Cannot scale the map: it or the model is constant within the mask.")

    correlation = torch.corrcoef(torch.stack([xmap_masked, model_masked]))[0, 1]
    if correlation <= 0:
        warnings.warn(
            f"Map and model density are not positively correlated within the mask "
            f"(r = {correlation:.3f}); the input structure may be misplaced relative to the map."
        )

    scaling_factor = model_std / xmap_std
    return scaling_factor * (xmap_array - xmap_masked.mean()) + model_masked.mean()


def to_f_density(real_map: torch.Tensor) -> torch.Tensor:
    """FFT a density map.

    Pads the map to odd shape before FFT to avoid half-sample ambiguity"""
    # f_density
    # pad map to odd shape
    map_shape = real_map.shape[-3:]
    pad_amount = [int(dim % 2 == 0) for dim in map_shape]
    if pad_amount != [0, 0, 0]:
        real_map = F.pad(real_map, (0, pad_amount[2], 0, pad_amount[1], 0, pad_amount[0]))

    f_map = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(real_map, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
    return f_map


def to_density(f_map: torch.Tensor) -> torch.Tensor:
    """Inverse FFT a density map."""
    # density
    return torch.real(
        torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(f_map, dim=(-3, -2, -1)),
                dim=(-3, -2, -1),
            ),
            dim=(-3, -2, -1),
        )
    )


def radial_hamming_3d(f_mag, cutoff_radius):
    """3D radial Hamming filter in Fourier space.

    Parameters
    ----------
    f_mag : Tensor
        Frequency magnitudes from FFT.
    cutoff_radius : float
        Frequency cutoff in same units as frequency coordinates.

    Returns
    -------
    Tensor
        3D tensor containing the Hamming filter.
    """
    filter = torch.zeros_like(f_mag)

    mask = f_mag <= cutoff_radius

    r_scaled = f_mag[mask] / cutoff_radius  # Scale to [0,1]
    hamming_vals = 0.54 + 0.46 * torch.cos(torch.pi * r_scaled)

    filter[mask] = hamming_vals

    return filter
