"""Tests for shared synthetic-data generation utilities."""

from pathlib import Path

import pytest
import torch
from sampleworks.synthetic.synthetic_utils import (
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
)


@pytest.mark.parametrize("n_jobs", [-2, -1, 2, 8])
def test_resolve_parallel_jobs_serializes_cuda_work(n_jobs: int) -> None:
    """CUDA requests that imply multiple processes are clamped to one worker."""
    assert resolve_parallel_jobs(torch.device("cuda:3"), n_jobs) == 1


@pytest.mark.parametrize("n_jobs", [-2, -1, 1, 2, 8])
def test_resolve_parallel_jobs_preserves_cpu_parallelism(n_jobs: int) -> None:
    """CPU calculations retain the requested joblib parallelism."""
    assert resolve_parallel_jobs(torch.device("cpu"), n_jobs) == n_jobs


def test_resolve_parallel_jobs_preserves_single_cuda_worker() -> None:
    """An explicitly serial CUDA request does not need adjustment."""
    assert resolve_parallel_jobs("cuda", 1) == 1


def test_load_structure_overrides_b_factors(resources_dir: Path) -> None:
    """A flat B-factor override replaces every retained input value."""
    atom_array = load_structure_for_synthetic_reward(
        resources_dir / "6b8x" / "6b8x_final.pdb",
        occupancy_mode="default",
        occupancy_values=[],
        strip_hydrogens=True,
        strip_waters=True,
        strip_ligands=True,
        selection="chain A",
        b_factor=100.0,
    )

    assert atom_array is not None
    assert (atom_array.b_factor == 100.0).all()


def test_load_structure_drops_zero_occupancy_atoms_after_assignment(
    resources_dir: Path,
) -> None:
    """Conformers assigned zero occupancy are absent from the prepared structure."""
    atom_array = load_structure_for_synthetic_reward(
        resources_dir / "6b8x" / "6b8x_final.pdb",
        occupancy_mode="custom",
        occupancy_values=[0.5, 0.5],
        selection="chain A",
    )

    assert atom_array is not None
    assert (atom_array.occupancy > 0.0).all()
    assert set(atom_array.altloc_id) - {"", ".", " ", "?"} == {"A", "B"}


@pytest.mark.parametrize("b_factor", [-1.0, float("inf"), float("nan")])
def test_load_structure_rejects_invalid_b_factor(
    resources_dir: Path, b_factor: float
) -> None:
    """B-factor overrides must be finite and non-negative."""
    with pytest.raises(ValueError, match="B-factor must be finite and non-negative"):
        load_structure_for_synthetic_reward(
            resources_dir / "6b8x" / "6b8x_final.pdb",
            occupancy_mode="default",
            occupancy_values=[],
            b_factor=b_factor,
        )
