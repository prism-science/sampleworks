"""Tests for chunked atom dilation on the PyTorch (non-CUDA-kernel) density path."""

import math

import pytest
import torch
from sampleworks.core.forward_models.xray.real_space_density import dilate_points_torch
from torch.utils._python_dispatch import TorchDispatchMode


GRID_SHAPE = (12, 12, 12)
VOXEL_SIZE = 0.5
RMAX = 1.5
RSTEP = 0.05
UNCHUNKED = 2**40


class LargestTensorMode(TorchDispatchMode):
    """Record the element count of the largest tensor any operator allocates."""

    def __init__(self) -> None:
        super().__init__()
        self.largest_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run ``func`` and record the size of its tensor outputs.

        Parameters
        ----------
        func
            The ATen operator.
        types
            Tensor subclasses involved in the call.
        args
            Positional arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        Any
            The operator's output.
        """
        output = func(*args, **(kwargs or {}))
        # Views (e.g. slicing an input) share storage, so they allocate nothing.
        if func.is_view:
            return output
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        for tensor in outputs:
            if isinstance(tensor, torch.Tensor):
                self.largest_numel = max(self.largest_numel, tensor.numel())
        return output


def dilation_inputs(batch_size: int, n_atoms: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random atoms on a periodic cubic grid, with every fifth atom inactive.

    Parameters
    ----------
    batch_size
        Number of structures.
    n_atoms
        Atoms per structure.
    seed
        Random seed.

    Returns
    -------
    dict[str, torch.Tensor]
        ``coordinates`` (batch, atoms, 3) in grid units, ``active`` and ``occupancies``
        (batch, atoms), and ``radial_densities`` (batch, atoms, n_radial); coordinates
        and occupancies require gradients.
    """
    generator = torch.Generator().manual_seed(seed)
    n_radial = round(RMAX / RSTEP) + 1
    active = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    active[:, ::5] = False
    return {
        "coordinates": (
            torch.rand(batch_size, n_atoms, 3, generator=generator) * GRID_SHAPE[0]
        ).requires_grad_(),
        "active": active,
        "occupancies": torch.rand(batch_size, n_atoms, generator=generator).requires_grad_(),
        "radial_densities": torch.rand(batch_size, n_atoms, n_radial, generator=generator),
    }


def dilate(inputs: dict[str, torch.Tensor], max_atom_voxel_pairs: int) -> torch.Tensor:
    """Dilate ``inputs`` onto ``GRID_SHAPE`` with a given chunk budget.

    Parameters
    ----------
    inputs
        Output of ``dilation_inputs``.
    max_atom_voxel_pairs
        Chunk budget passed to ``dilate_points_torch``.

    Returns
    -------
    torch.Tensor
        Density of shape (batch, *GRID_SHAPE).
    """
    return dilate_points_torch(
        inputs["coordinates"],
        inputs["active"],
        inputs["occupancies"],
        torch.full((3,), RMAX / VOXEL_SIZE),
        inputs["radial_densities"],
        RSTEP,
        RMAX,
        VOXEL_SIZE * torch.eye(3),
        GRID_SHAPE,
        max_atom_voxel_pairs=max_atom_voxel_pairs,
    )


def n_nearby_voxels() -> int:
    """Number of voxels in the search box around each atom.

    Returns
    -------
    int
        Voxel count of the cube reaching ``RMAX`` from the atom's voxel.
    """
    return (2 * math.ceil(RMAX / VOXEL_SIZE) + 1) ** 3


@pytest.mark.parametrize("atoms_per_chunk", [1, 3, 8])
def test_chunking_does_not_change_density_or_gradients(atoms_per_chunk):
    """Chunked dilation matches a single chunk, including gradients, for uneven chunks.

    Parameters
    ----------
    atoms_per_chunk
        Atoms per chunk; 3 does not divide the atom count, so the last chunk is short.
    """
    batch_size, n_atoms = 2, 10
    budget = atoms_per_chunk * batch_size * n_nearby_voxels()

    unchunked_inputs = dilation_inputs(batch_size, n_atoms)
    unchunked = dilate(unchunked_inputs, UNCHUNKED)
    unchunked.square().sum().backward()

    chunked_inputs = dilation_inputs(batch_size, n_atoms)
    chunked = dilate(chunked_inputs, budget)
    chunked.square().sum().backward()

    assert unchunked.abs().sum() > 0
    torch.testing.assert_close(chunked, unchunked)
    for name in ("coordinates", "occupancies"):
        torch.testing.assert_close(chunked_inputs[name].grad, unchunked_inputs[name].grad)


def test_chunk_of_only_inactive_atoms_adds_nothing():
    """A chunk with no active atoms contributes no density and no gradient."""
    inputs = dilation_inputs(batch_size=1, n_atoms=4)
    inputs["active"][:] = False
    density = dilate(inputs, n_nearby_voxels())
    density.sum().backward()

    coordinate_gradient = inputs["coordinates"].grad
    assert torch.count_nonzero(density) == 0
    assert coordinate_gradient is not None
    assert torch.count_nonzero(coordinate_gradient) == 0


def test_backward_keeps_no_per_voxel_tensors():
    """Autograd saves less than one byte per atom-voxel pair, i.e. only the inputs, so
    memory held for backward does not grow with the search box."""
    batch_size, n_atoms = 2, 160
    saved_storage_bytes = {}

    def record_storage(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        saved_storage_bytes[storage.data_ptr()] = storage.nbytes()
        return tensor

    inputs = dilation_inputs(batch_size, n_atoms)
    with torch.autograd.graph.saved_tensors_hooks(record_storage, lambda tensor: tensor):
        density = dilate(inputs, 4 * batch_size * n_nearby_voxels())
    density.sum().backward()

    assert sum(saved_storage_bytes.values()) < batch_size * n_atoms * n_nearby_voxels()


def test_largest_intermediate_does_not_grow_with_atom_count():
    """With a fixed chunk budget, the largest tensor in the forward and backward passes
    is the same for 8x the atoms, and stays within three values per atom-voxel pair."""
    batch_size = 2
    budget = 4 * batch_size * n_nearby_voxels()

    largest = {}
    for n_atoms in (20, 160):
        inputs = dilation_inputs(batch_size, n_atoms)
        with LargestTensorMode() as mode:
            dilate(inputs, budget).sum().backward()
        largest[n_atoms] = mode.largest_numel

    assert largest[160] == largest[20]
    assert largest[160] <= max(3 * budget, batch_size * math.prod(GRID_SHAPE))
