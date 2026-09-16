"""
Shared fixtures for `tests/eval/`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from sampleworks.eval.occupancy_utils import occupancy_to_str


_RESOURCES = Path(__file__).resolve().parents[1] / "resources" / "1vme"
_REAL_CIF = _RESOURCES / "1VME_single_001_density_input.cif"
_MAP_RESOLUTION = 2.0

# Occupancy pairs used to produce distinct `(protein, occ_key)` groups.
_GROUP_OCCUPANCIES: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.25, 0.75),
)

DEFAULT_SELECTIONS: tuple[str, ...] = (
    "chain A and resi 326-339",
    "chain A and resi 326-332",
)


def _link(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``, creating parent dirs and replacing any
    existing link. Uses a symlink so the CCP4 is not duplicated per
    trial."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _populate(
    root: Path,
    n_groups: int,
    trials_per_group: int,
    selections: Sequence[str],
    base_map: Path,
) -> argparse.Namespace:
    assert _REAL_CIF.exists(), _REAL_CIF
    assert 1 <= n_groups <= len(_GROUP_OCCUPANCIES)
    assert trials_per_group >= 1
    assert len(selections) >= 1

    group_strs = [occupancy_to_str(A=a, B=b) for a, b in _GROUP_OCCUPANCIES[:n_groups]]

    inputs = root / "inputs"
    base_map_dir_rel = Path("1vme_fixture_dir")
    base_map_dir = inputs / base_map_dir_rel

    for s in group_strs:
        _link(base_map, base_map_dir / f"{s}.ccp4")
        _link(_REAL_CIF, base_map_dir / f"{s}.cif")
    # Default ``occ_list`` in get_reference_structure_coords looks these up.
    for s in ("1.0occA", "1.0occB"):
        _link(_REAL_CIF, base_map_dir / f"{s}.cif")

    configs_csv = inputs / "protein_configs.csv"
    configs_csv.parent.mkdir(parents=True, exist_ok=True)
    configs_csv.write_text(
        "protein,base_map_dir,selection,resolution,map_pattern,structure_pattern\n"
        f"1vme,{base_map_dir_rel},{';'.join(selections)},{_MAP_RESOLUTION},"
        "{occ_str}.ccp4,{occ_str}.cif\n"
    )

    # grid_search_results tree: protein_dir/model_dir/scaler_dir/trial_dir/refined.cif
    results_root = root / "grid_search_results"
    results_root.mkdir(parents=True, exist_ok=True)
    for s in group_strs:
        protein_dir = results_root / f"1vme_{s}"
        scaler_dir = protein_dir / "BOLTZ_2_bench" / "fk_steering"
        for i in range(trials_per_group):
            trial_dir = scaler_dir / f"ens2_gw{1.0 + i}"
            _link(_REAL_CIF, trial_dir / "refined.cif")

    # The Namespace the script's ``main`` consumes. ``selections`` rides along
    # for test assertions (the script ignores attributes it doesn't read).
    return argparse.Namespace(
        grid_search_results_path=results_root,
        grid_search_inputs_path=inputs,
        protein_configs_csv=configs_csv,
        depth=4,
        target_filename="refined.cif",
        n_jobs=1,  # keep joblib in-process so test monkeypatches reach the work
        selections=tuple(selections),
    )


@pytest.fixture(scope="session")
def synthetic_base_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Generate a small synthetic CCP4 map from the reference CIF, once per session
    """
    import numpy as np
    import torch
    from atomworks.io.parser import parse
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import UnitCell
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
        GridParameters,
        Resolution,
        XMap,
    )
    from sampleworks.utils.atom_array_utils import remove_atoms_with_any_nan_coords
    from sampleworks.utils.density_utils import compute_density_from_atomarray
    from sampleworks.utils.structure_utils import get_asym_unit_from_structure

    atom_array = remove_atoms_with_any_nan_coords(
        get_asym_unit_from_structure(parse(_REAL_CIF, ccd_mirror_path=None))
    )
    coords = np.asarray(atom_array.coord)  # AtomArray (n_atoms, 3) or stack (n_models, n_atoms, 3)
    if coords.ndim == 3:
        coords = coords.reshape(-1, 3)
    coords = coords[np.isfinite(coords).all(axis=1)]  # (x, y, z) per atom
    voxel = _MAP_RESOLUTION / 4.0
    grid_shape = np.ceil((coords.max(axis=0) + 5.0) / voxel).astype(int)  # (nx, ny, nz)
    abc = grid_shape * voxel
    xmap = XMap(
        np.zeros(grid_shape[::-1], dtype=np.float32),  # (nz, ny, nx) array ordering
        grid_parameters=GridParameters(voxelspacing=voxel),
        unit_cell=UnitCell(
            a=float(abc[0]),
            b=float(abc[1]),
            c=float(abc[2]),
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            space_group="P1",
        ),
        resolution=Resolution(high=_MAP_RESOLUTION, low=1000.0),
    )
    density, _ = compute_density_from_atomarray(
        atom_array,
        xmap=xmap,
        em_mode=False,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    xmap.array = density.cpu().numpy()
    out_path = tmp_path_factory.mktemp("rscc_synthetic_map") / "base.ccp4"
    xmap.tofile(str(out_path))
    return out_path


@pytest.fixture
def rscc_fixture_factory(
    tmp_path: Path,
    synthetic_base_map: Path,
) -> Callable[..., argparse.Namespace]:
    """Return a factory that builds an RSCC fixture under ``tmp_path``.

    Parameters:

    - ``n_groups`` (default 2): number of occupancy groups to create.
    - ``trials_per_group`` (default 2): trial subdirectories per group.
    - ``selections`` (default :data:`DEFAULT_SELECTIONS`): selection strings.

    Returns the ``argparse.Namespace`` the script's ``main`` consumes, with a
    ``selections`` attribute attached for assertions.
    """

    def _factory(
        *,
        n_groups: int = 2,
        trials_per_group: int = 2,
        selections: Sequence[str] = DEFAULT_SELECTIONS,
    ) -> argparse.Namespace:
        return _populate(tmp_path, n_groups, trials_per_group, selections, synthetic_base_map)

    return _factory
