"""Tests for shared synthetic-data generation utilities."""

from pathlib import Path

import pytest
import torch
from sampleworks.synthetic.synthetic_utils import (
    BatchRowForMTZ,
    load_structure_for_synthetic_reward,
    resolve_crystal_metadata,
    resolve_parallel_jobs,
    rfree_flags,
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


@pytest.mark.parametrize("b_factor", [1.0, 37.5, 100.0])
def test_load_structure_overrides_b_factors(resources_dir: Path, b_factor: float) -> None:
    """A flat B-factor override replaces every retained input value."""
    atom_array = load_structure_for_synthetic_reward(
        resources_dir / "6b8x" / "6b8x_final.pdb",
        occupancy_mode="default",
        occupancy_values=[],
        strip_hydrogens=True,
        strip_waters=True,
        strip_ligands=True,
        selection="chain A",
        b_factor=b_factor,
    )

    assert atom_array is not None
    assert (atom_array.b_factor == b_factor).all()


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
def test_load_structure_rejects_invalid_b_factor(resources_dir: Path, b_factor: float) -> None:
    """B-factor overrides must be finite and non-negative."""
    with pytest.raises(ValueError, match="B-factor must be finite and non-negative"):
        load_structure_for_synthetic_reward(
            resources_dir / "6b8x" / "6b8x_final.pdb",
            occupancy_mode="default",
            occupancy_values=[],
            b_factor=b_factor,
        )


def test_resolve_crystal_metadata_prefers_the_row_over_the_file(resources_dir: Path) -> None:
    """Row overrides win; whatever the row leaves unset comes from the structure file."""
    import gemmi

    source = resources_dir / "1vme" / "1vme_final.cif"
    file_cell, file_hm = (lambda m: (m.cell, m.spacegroup_hm))(gemmi.read_structure(str(source)))

    cell, hm = resolve_crystal_metadata(BatchRowForMTZ(filename=source.name), source)
    assert cell.parameters == file_cell.parameters and hm == file_hm

    # Overrides derived from the file's own values, so they differ by construction.
    other_cell = tuple(x + 1.0 for x in file_cell.parameters[:3]) + file_cell.parameters[3:]
    other_hm = "P 1" if file_hm != "P 1" else "P 2 2 2"
    row = BatchRowForMTZ.from_dict(
        {
            "filename": source.name,
            "unit_cell": ":".join(f"{x:g}" for x in other_cell),
            "space_group": other_hm,
        }
    )
    cell, hm = resolve_crystal_metadata(row, source)
    assert cell.parameters == pytest.approx(other_cell)
    assert hm == other_hm


@pytest.mark.parametrize("ccp4_convention", [False, True], ids=["phenix", "ccp4"])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_rfree_flags_match_reciprocalspaceship(seed: int, ccp4_convention: bool) -> None:
    """Same label and the same flag on every reflection as rs.utils.add_rfree."""
    import reciprocalspaceship as rs

    n = 500
    reference = rs.DataSet({"H": range(n), "K": 0, "L": 0}).set_index(["H", "K", "L"])
    reference = rs.utils.add_rfree(
        reference, fraction=0.1, seed=seed, ccp4_convention=ccp4_convention
    )

    label, flags = rfree_flags(n, 0.1, seed=seed, ccp4_convention=ccp4_convention)

    assert [label] == list(reference.columns)
    assert flags.tolist() == reference[label].to_numpy().astype(float).tolist()
