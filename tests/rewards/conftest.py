"""Reward-specific pytest fixtures.

Fixtures here are scoped to the reward test suite (`tests/rewards/`) so they don't
pollute the global fixture namespace. They depend on cross-cutting fixtures defined in
the parent `tests/conftest.py` (`device`, `resources_dir`, `density_map_1vme`), which
pytest resolves up the conftest hierarchy automatically.

`density_map_1vme` deliberately stays in the top-level conftest because
`tests/utils/test_density_utils.py` also consumes it; a fixture lives at the lowest
common ancestor of everything that uses it.

The two reward families get their target data differently. The density fixtures read
committed files (a carved .ccp4 and its matching recentered cif). The SF fixtures
generate their (cif, mtz) pair per session via `generate_synthetic_sf.py`.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from atomworks.io.parser import parse
from biotite.structure import AtomArray
from jaxtyping import Float
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
        XMap,
    )
    from sampleworks.core.rewards.real_space_density import RealSpaceRewardFunction
    from sampleworks.core.rewards.structure_factor import StructureFactorRewardFunction


@pytest.fixture(scope="session")
def structure_1vme_density(resources_dir: Path) -> dict:
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Structure not found at {cif_path}")
    return parse(cif_path, ccd_mirror_path=None)


@pytest.fixture(scope="session")
def reward_function_1vme(
    density_map_1vme: "XMap", device: torch.device
) -> "RealSpaceRewardFunction":
    """Real-space density reward from the carved 1vme map, in X-ray (non-EM) mode."""
    from sampleworks.core.rewards.real_space_density import (
        RealSpaceRewardFunction,
        setup_scattering_params,
    )

    params = setup_scattering_params(em_mode=False, device=device)
    rf = RealSpaceRewardFunction(density_map_1vme, params, torch.tensor([1], device=device))
    return rf


@pytest.fixture(scope="session")
def test_coordinates_1vme(
    structure_1vme_density: dict, device: torch.device
) -> tuple[Float[Tensor, "n_atoms 3"], AtomArray]:
    """Atomic coordinates and the atom array (asym_unit) for the density model,
    zero-occupancy atoms dropped."""
    atom_array = structure_1vme_density["asym_unit"]

    # Handle both AtomArray and AtomArrayStack
    if hasattr(atom_array, "stack_depth"):
        # AtomArrayStack - take first model
        atom_array = atom_array[0]

    mask = atom_array.occupancy > 0
    atom_array = atom_array[mask]
    coords = torch.from_numpy(atom_array.coord).to(device=device, dtype=torch.float32)
    return coords, atom_array


@pytest.fixture(scope="session")
def sf_1vme_cif_and_mtz_paths(
    resources_dir: Path, tmp_path_factory: pytest.TempPathFactory, device: torch.device
) -> tuple[Path, Path]:
    """Generate the synthetic SF data and the exact structure used for the generation,
    returns the ``(cif, mtz)`` paths.

    Mirrors the ``--structure`` branch of ``generate_synthetic_sf.main()``, so these
    fixtures exercise exactly what the CLI produces. ``--save-structure`` writes back the
    post-selection and occupancy modifed model, which is the structure actually used to
    generate the synthetic SF data.

    Specifically, we select for the chain A of 1vme with altloc occupancies forced to be
    0.5/0.5 and H + waters stripped, at 1.8 A resolution, with bulk solvent and default
    scales. The resulted mtz file carries both Fprotein/SIGFprotein/PHIFprotein and
    Ftotal/SIGFtotal/PHIFtotal.
    """
    from sampleworks.synthetic.generate_synthetic_sf import (
        _process_single_row,
        BatchRowForMTZ,
    )

    source_cif = "1vme_final.cif"
    source_dir = resources_dir / "1vme"
    if not (source_dir / source_cif).exists():
        pytest.skip(f"Source structure not found at {source_dir / source_cif}")

    resolution = 1.8
    output_dir = tmp_path_factory.mktemp("sf_1vme")
    _process_single_row(
        row=BatchRowForMTZ(
            filename=source_cif,
            selection="chain A",
            occupancy_values=[0.5, 0.5],  # assigned to altlocs in sorted order, A then B
        ),
        base_dir=source_dir,
        output_dir=output_dir,
        resolution=resolution,
        scattering_factor_mode="xray",
        occupancy_mode="custom",
        test_fraction=0.1,  # needed for test_inverted_testset_value_warns
        seed=0,  # R-free flag assignment only
        device=device,
        strip_hydrogens=True,
        strip_waters=True,
        simulate_solvent_and_scale=True,
        save_structure=True,
    )

    # Following the default file names in the generate_synthetic_sf.py script:
    # `{stem}_{resolution:.2f}A.mtz`, and the saved structure as `{stem}_sf_input.cif`.
    # Ideally we want to modify _process_single_row to return the paths directly in case
    # the naming defaults changed or accept naming parameters.
    # Because _process_single_row logs and returns on failure instead of raising errors,
    # missing output is the only way to check generation success.
    stem = Path(source_cif).stem
    cif_path = output_dir / f"{stem}_sf_input.cif"
    mtz_path = output_dir / f"{stem}_{resolution:.2f}A.mtz"
    missing = [p.name for p in (cif_path, mtz_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"generate_synthetic_sf wrote no {missing} to {output_dir}; see the "
            f"generator's logged error for the cause"
        )
    return cif_path, mtz_path


@pytest.fixture(scope="session")
def mtz_path_1vme(sf_1vme_cif_and_mtz_paths: tuple[Path, Path]) -> Path:
    """Synthetic 1.8 A target: Fprotein and Ftotal sets, each with its SIGF and phase."""
    _, mtz_path = sf_1vme_cif_and_mtz_paths
    return mtz_path


@pytest.fixture(scope="session")
def structure_1vme_sf(sf_1vme_cif_and_mtz_paths: tuple[Path, Path]) -> AtomArray:
    """The exact chain-A model with altlocs in the P2_1 crystal frame that the mtz was
    computed from."""
    from sampleworks.utils.atom_array_utils import load_structure_with_altlocs

    cif_path, _ = sf_1vme_cif_and_mtz_paths
    return load_structure_with_altlocs(cif_path)


@pytest.fixture(scope="session")
def test_coordinates_1vme_sf(
    structure_1vme_sf: AtomArray, device: torch.device
) -> tuple[Float[Tensor, "n_atoms 3"], AtomArray]:
    """Atomic coordinates and the atom array (asym_unit) which for the SF reward is the
    same atom array as the structure_1vme_sf.

    Exists for the ``_REWARD_BUNDLES`` mapping in ``test_reward_function_contract.py``,
    which resolves one ``(coords, atom_array)`` fixture per reward so every reward is
    exercised through the same shape. SF-specific tests that need only the atom array take
    ``structure_1vme_sf`` directly.
    """
    # No occupancy>0 filter here (unlike the density fixture) as SFC topology is fixed.
    coords = torch.from_numpy(structure_1vme_sf.coord).to(device=device, dtype=torch.float32)
    return coords, structure_1vme_sf


@pytest.fixture(scope="session")
def reward_function_1vme_sf(
    mtz_path_1vme: Path,
    structure_1vme_sf: AtomArray,
    device: torch.device,
) -> "StructureFactorRewardFunction":
    """Structure factor reward (|Fprotein| only) prepared using the 1vme structure's topology."""
    from sampleworks.core.rewards.protocol import RewardInputs
    from sampleworks.core.rewards.structure_factor import StructureFactorRewardFunction

    # normalize_amplitude=True scores normalized E-values (|Ec| vs sfc.Eo), which are
    # unit-variance per resolution shell, so the MSE can be tested on an absolute scale.
    reward_function = StructureFactorRewardFunction(
        mtz_path_1vme,
        expcolumns=["Fprotein", "SIGFprotein"],
        normalize_amplitude=True,
    )
    reward_function.prepare(
        RewardInputs.from_atom_array(structure_1vme_sf, ensemble_size=1, device=device),
        device=device,
    )
    return reward_function
