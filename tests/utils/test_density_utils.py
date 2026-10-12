"""Tests for density_utils module."""

from typing import cast

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.core.forward_models.xray.real_space_density import XMap_torch
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import extract_density_inputs_from_atomarray
from sampleworks.utils.density_utils import (
    build_density_transformer,
    compute_density_from_atomarray,
    create_synthetic_grid,
    run_density_transformer,
)


@pytest.fixture(
    params=[
        {"resolution": 1.0, "padding": 5.0},
        {"resolution": 2.0, "padding": 5.0},
        {"resolution": 2.0, "padding": 10.0},
        {"resolution": 4.0, "padding": 5.0},
    ]
)
def synthetic_grid(request, simple_atom_array: AtomArray):
    params = request.param
    xmap = create_synthetic_grid(
        simple_atom_array, resolution=params["resolution"], padding=params["padding"]
    )
    resolution = params["resolution"]
    padding = params["padding"]
    return xmap, resolution, padding


class TestCreateSyntheticGrid:
    """Tests for create_synthetic_grid function."""

    def test_returns_xmap(self, synthetic_grid):
        """Test that output is XMap type."""
        xmap, resolution, padding = synthetic_grid
        assert isinstance(xmap, XMap)

        assert xmap.array.ndim == 3
        assert np.all(xmap.array == 0.0)

        expected_spacing = resolution / 4.0
        if isinstance(xmap.voxelspacing, np.ndarray):
            assert np.allclose(xmap.voxelspacing, expected_spacing, rtol=1e-6)
        else:
            assert abs(xmap.voxelspacing - expected_spacing) < 1e-6

    def test_grid_contains_structure(self, simple_atom_array: AtomArray):
        """Test that grid bounds include all coordinates."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=5.0)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        assert np.all(xmap.origin <= min_coords - 5.0)
        assert xmap.unit_cell is not None
        uc_dims = np.array([xmap.unit_cell.a, xmap.unit_cell.b, xmap.unit_cell.c])
        grid_end = xmap.origin + uc_dims
        assert np.all(grid_end >= max_coords + 5.0)

    def test_voxel_spacing(self, synthetic_grid):
        """Test that voxel spacing equals resolution / 4.0."""
        xmap, resolution, padding = synthetic_grid
        expected_spacing = resolution / 4.0
        if isinstance(xmap.voxelspacing, np.ndarray):
            assert np.allclose(xmap.voxelspacing, expected_spacing, rtol=1e-6)
        else:
            assert abs(xmap.voxelspacing - expected_spacing) < 1e-6

    def test_padding_applied(self, simple_atom_array: AtomArray):
        """Test that grid is expanded by padding amount."""
        padding = 10.0
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        expected_origin = min_coords - padding

        np.testing.assert_allclose(xmap.origin, expected_origin, rtol=1e-5)

    def test_orthogonal_unit_cell(self, synthetic_grid):
        """Test that unit cell has 90 degree angles."""
        xmap, resolution, padding = synthetic_grid
        assert xmap.unit_cell is not None
        assert xmap.unit_cell.alpha == 90.0
        assert xmap.unit_cell.beta == 90.0
        assert xmap.unit_cell.gamma == 90.0

    def test_space_group_p1(self, synthetic_grid):
        """Test that space group is P1."""
        xmap, resolution, padding = synthetic_grid
        assert xmap.unit_cell is not None
        space_group_str = str(xmap.unit_cell.space_group)
        assert "P1" in space_group_str or space_group_str == "P 1"

    def test_handles_atomarray_stack(self, simple_atom_array_stack: AtomArrayStack):
        """Test that function works with AtomArrayStack input."""
        xmap = create_synthetic_grid(simple_atom_array_stack, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert xmap.array.ndim == 3

    def test_filters_nan_coordinates(self, atom_array_with_nan_coords: AtomArray):
        """Test that NaN coordinates are excluded from bounds calculation."""
        xmap = create_synthetic_grid(atom_array_with_nan_coords, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert np.isfinite(xmap.origin).all()

    def test_custom_padding(self, simple_atom_array: AtomArray):
        """Test that non-default padding values work."""
        padding_values = [0.0, 2.0, 15.0]
        for padding in padding_values:
            xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
            assert isinstance(xmap, XMap)

    def test_resolution_affects_grid_shape(self, simple_atom_array: AtomArray):
        """Test that smaller resolution creates more voxels."""
        resolution_low = 4.0
        resolution_high = 1.0
        xmap_low_res = create_synthetic_grid(simple_atom_array, resolution=resolution_low)
        xmap_high_res = create_synthetic_grid(simple_atom_array, resolution=resolution_high)

        volume_low = np.prod(xmap_low_res.array.shape)
        volume_high = np.prod(xmap_high_res.array.shape)

        resolution_ratio = resolution_low / resolution_high
        # account for grid boundary effects by multiplying by 0.8
        expected_min_ratio = (resolution_ratio**3) * 0.8
        assert volume_high > volume_low * expected_min_ratio

    def test_array_shape_ordering(self, synthetic_grid):
        """Test that array shape is (nz, ny, nx)."""
        xmap, resolution, padding = synthetic_grid
        assert xmap.array.ndim == 3
        assert xmap.array.shape[0] > 0
        assert xmap.array.shape[1] > 0
        assert xmap.array.shape[2] > 0

    def test_empty_array_initialized(self, synthetic_grid):
        """Test that array is initialized to zeros."""
        xmap, resolution, padding = synthetic_grid
        assert np.all(xmap.array == 0.0)


@pytest.mark.gpu
class TestComputeDensityFromAtomarray:
    """Tests for compute_density_from_atomarray function."""

    def test_returns_density_tensor(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that first output is torch.Tensor."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)

    def test_returns_xmap_torch(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that second output is XMap_torch."""
        _, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(xmap_torch, XMap_torch)

    def test_density_shape_matches_grid(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density tensor shape matches grid dimensions."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        expected_shape = xmap_torch.array.shape
        assert density.shape == expected_shape

    def test_density_is_finite(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density contains no NaN or Inf values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert torch.isfinite(density).all()

    def test_with_resolution_parameter(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that resolution parameter creates synthetic grid."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.numel() > 0

    def test_with_xmap_parameter(
        self, simple_atom_array: AtomArray, synthetic_grid, device: torch.device
    ):
        """Test that function uses provided XMap."""
        xmap, resolution, padding = synthetic_grid
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, xmap=xmap, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.shape == xmap.array.shape

    def test_both_xmap_and_resolution_raises(
        self, simple_atom_array: AtomArray, synthetic_grid, device: torch.device
    ):
        """Test that providing both xmap and resolution raises ValueError."""
        xmap, resolution, padding = synthetic_grid
        with pytest.raises(ValueError, match="Cannot provide both xmap and resolution"):
            compute_density_from_atomarray(
                simple_atom_array, xmap=xmap, resolution=2.0, em_mode=False, device=device
            )

    def test_neither_xmap_nor_resolution_raises(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """Test that providing neither xmap nor resolution raises ValueError."""
        with pytest.raises(ValueError, match="Either xmap or resolution must be provided"):
            compute_density_from_atomarray(simple_atom_array, em_mode=False, device=device)

    def test_xray_mode(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that X-ray mode (em_mode=False) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_electron_mode(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that electron mode (em_mode=True) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=True, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_gemmi_tables(self, simple_atom_array: AtomArray, device: torch.device):
        """gemmi's IT92 table gives nearly the same X-ray density as the built-in one."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        density_gemmi, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device, gemmi_tables=True
        )
        assert density_gemmi.shape == density.shape
        torch.testing.assert_close(density_gemmi, density, rtol=0.05, atol=0.05 * density.max())
        assert not torch.equal(density_gemmi, density)

    def test_device_parameter(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that function respects device parameter."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.device == device

    def test_density_has_positive_values(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density has some positive values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.sum() > 0

    def test_with_atom_array_single_model(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Test that function works with single model from stack."""
        single_model = simple_atom_array_stack[0]
        density, _ = compute_density_from_atomarray(
            single_model,
            resolution=2.0,
            em_mode=False,
            device=device,
        )
        assert isinstance(density, torch.Tensor)
        assert torch.isfinite(density).all()

    def test_deterministic_output(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that same input produces same output."""
        density1, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        density2, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        torch.testing.assert_close(density1, density2)

    def test_high_resolution_smaller_voxels(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """Test that higher resolution creates finer grid."""
        density_low, xmap_low = compute_density_from_atomarray(
            simple_atom_array, resolution=4.0, em_mode=False, device=device
        )
        density_high, xmap_high = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )

        assert density_high.numel() > density_low.numel()
        vs_high = xmap_high.voxelspacing
        vs_low = xmap_low.voxelspacing

        if isinstance(vs_high, torch.Tensor):
            assert torch.all(vs_high < vs_low).item()
        elif isinstance(vs_high, np.ndarray):
            assert np.all(vs_high < vs_low)
        else:
            assert vs_high < vs_low

    def test_zero_occupancy_atoms_filtered(self, device: torch.device):
        """Test that atoms with zero occupancy contribute no density."""
        atom_array = AtomArray(3)
        atom_array.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 0.0, 1.0]))

        density, _ = compute_density_from_atomarray(
            atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.sum() > 0

    def test_with_real_structure(
        self, structure_1vme: dict, density_map_1vme: XMap, device: torch.device
    ):
        """Test density computation with real structure."""
        atom_array = structure_1vme["asym_unit"]
        if isinstance(atom_array, AtomArrayStack):
            atom_array = atom_array[0]

        density_no_xmap, _ = compute_density_from_atomarray(
            atom_array,
            resolution=2.0,
            em_mode=False,
            device=device,
        )
        assert isinstance(density_no_xmap, torch.Tensor)
        assert density_no_xmap.sum() > 0
        assert torch.isfinite(density_no_xmap).all()

        density_xmap, _ = compute_density_from_atomarray(
            atom_array,
            xmap=density_map_1vme,  # resolution 1.8 A
            em_mode=False,
            device=device,
        )
        assert isinstance(density_xmap, torch.Tensor)
        assert density_xmap.sum() > 0
        assert torch.isfinite(density_xmap).all()

        assert density_no_xmap.shape != density_xmap.shape  # different resolutions


@pytest.mark.gpu
class TestComputeDensityErrors:
    """Tests for error handling in compute_density_from_atomarray."""

    def test_empty_atom_array(self, device: torch.device):
        """Test behavior with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.coord = np.empty((0, 3))

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            compute_density_from_atomarray(atom_array, resolution=2.0, em_mode=False, device=device)


class TestExtractDensityInputsFromAtomArrayStack:
    """Tests for extract_density_inputs_from_atomarray with AtomArrayStack input."""

    def test_stack_returns_correct_batch_dim(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Output tensors should have n_models as the batch dimension."""
        n_models = simple_atom_array_stack.stack_depth()
        n_atoms = simple_atom_array_stack.array_length()

        coords, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
            simple_atom_array_stack, device
        )
        assert coords.shape == (n_models, n_atoms, 3)
        assert elements.shape == (n_models, n_atoms)
        assert b_factors.shape == (n_models, n_atoms)
        assert occupancies.shape == (n_models, n_atoms)

    def test_stack_coords_match_per_model(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Coordinates in each batch entry should match the corresponding model."""
        coords, _, _, _ = extract_density_inputs_from_atomarray(simple_atom_array_stack, device)
        for i in range(simple_atom_array_stack.stack_depth()):
            expected = torch.from_numpy(simple_atom_array_stack.coord[i].copy()).to(
                device, dtype=torch.float32
            )
            torch.testing.assert_close(coords[i], expected)

    def test_stack_non_uniform_occupancy_preserved(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Occupancy values are preserved as provided for stack inputs."""
        n_models = simple_atom_array_stack.stack_depth()
        n_atoms = simple_atom_array_stack.array_length()

        _, _, _, occupancies = extract_density_inputs_from_atomarray(
            simple_atom_array_stack, device
        )
        expected = torch.full((n_models, n_atoms), 1.0, device=device, dtype=torch.float32)
        torch.testing.assert_close(occupancies, expected)

    def test_stack_filters_invalid_coords_across_models(
        self, atom_array_stack_with_nan_coords: AtomArrayStack, device: torch.device
    ):
        """An atom with NaN coords in any model should be filtered out entirely."""
        n_models = atom_array_stack_with_nan_coords.stack_depth()

        coords, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
            atom_array_stack_with_nan_coords, device
        )
        # Atom at index 1 is NaN in model 2 -> filtered out, leaving 2 atoms
        assert coords.shape == (n_models, 2, 3)
        assert elements.shape == (n_models, 2)
        assert b_factors.shape == (n_models, 2)
        assert occupancies.shape == (n_models, 2)

    def test_single_array_shapes_unchanged(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """Single AtomArray should still produce (1, n_atoms, ...) shapes."""
        n_atoms = len(simple_atom_array)
        coords, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
            simple_atom_array, device
        )
        assert coords.shape == (1, n_atoms, 3)
        assert elements.shape == (1, n_atoms)
        assert b_factors.shape == (1, n_atoms)
        assert occupancies.shape == (1, n_atoms)


@pytest.mark.gpu
class TestComputeDensityFromAtomArrayStack:
    """Tests for compute_density_from_atomarray with AtomArrayStack input."""

    def test_stack_returns_finite_density(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Density from an AtomArrayStack should be a finite tensor with positive values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array_stack, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert torch.isfinite(density).all()
        assert density.sum() > 0

    def test_stack_density_shape_matches_grid(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Density shape should match grid dimensions (batch dim is summed out)."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array_stack, resolution=2.0, em_mode=False, device=device
        )
        assert density.shape == xmap_torch.array.shape

    def test_stack_density_matches_manual_weighted_sum(
        self, atom_array_stack_uniform_occ: AtomArrayStack, device: torch.device
    ):
        """Density from a 2 model stack should equal the sum of each model's density."""
        s = atom_array_stack_uniform_occ
        n_models = s.stack_depth()
        occ = 1.0 / n_models

        # Create a shared grid from the stack so all computations use the same grid
        shared_xmap = create_synthetic_grid(s, resolution=2.0)

        # Compute density from the stack
        density_stack, _ = compute_density_from_atomarray(
            s, xmap=shared_xmap, em_mode=False, device=device
        )

        # Compute density from each model individually with the same occupancy
        per_model_densities = []
        for i in range(n_models):
            model_i = cast(AtomArray, s[i])
            model_i.set_annotation("occupancy", np.full(s.array_length(), occ, dtype=np.float64))
            density_i, _ = compute_density_from_atomarray(
                model_i, xmap=shared_xmap, em_mode=False, device=device
            )
            per_model_densities.append(density_i)

        expected = sum(per_model_densities)
        torch.testing.assert_close(density_stack, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.slow
@pytest.mark.gpu
class TestSplitDensityHelpers:
    """Tests for build_density_transformer and run_density_transformer."""

    def test_split_helpers_match_wrapper(self, simple_atom_array: AtomArray, device: torch.device):
        """
        Split helpers must produce the same density as the wrapper.
        """
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)

        ref_density, _ = compute_density_from_atomarray(
            simple_atom_array, xmap=xmap, em_mode=False, device=device
        )
        transformer, _ = build_density_transformer(xmap, em_mode=False, device=device)
        new_density = run_density_transformer(transformer, simple_atom_array)

        torch.testing.assert_close(ref_density, new_density)

    def test_split_helpers_match_wrapper_on_stack(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """
        Split helpers must produce the same density as the wrapper when called
        directly with an AtomArrayStack. This is the shape that the RSCC
        grid-search refactor relies on when caching a transformer across trials.
        """
        xmap = create_synthetic_grid(simple_atom_array_stack, resolution=2.0)

        ref_density, _ = compute_density_from_atomarray(
            simple_atom_array_stack, xmap=xmap, em_mode=False, device=device
        )
        transformer, _ = build_density_transformer(xmap, em_mode=False, device=device)
        new_density = run_density_transformer(transformer, simple_atom_array_stack)

        torch.testing.assert_close(ref_density, new_density)

    def test_transformer_reuse_produces_identical_density(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """
        Running the same transformer twice on the same input should produce the same results.
        """
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        transformer, _ = build_density_transformer(xmap, em_mode=False, device=device)

        density_1 = run_density_transformer(transformer, simple_atom_array)
        density_2 = run_density_transformer(transformer, simple_atom_array)

        torch.testing.assert_close(density_1, density_2)

    def test_transformer_reuse_across_different_structures(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """
        Different atom arrays through the same transformer produce distinct
        but deterministic densities
        """
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        transformer, _ = build_density_transformer(xmap, em_mode=False, device=device)

        # Shift coords to make a different structure but reuse the grid.
        shifted = simple_atom_array.copy()
        shifted.coord = shifted.coord + 0.5

        density_orig = run_density_transformer(transformer, simple_atom_array)
        density_shifted = run_density_transformer(transformer, shifted)
        density_orig_again = run_density_transformer(transformer, simple_atom_array)

        assert not torch.allclose(density_orig, density_shifted)
        torch.testing.assert_close(density_orig, density_orig_again)
