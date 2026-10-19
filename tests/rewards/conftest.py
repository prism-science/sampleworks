"""Reward-specific pytest fixtures.

Fixtures here are scoped to the reward test suite (`tests/rewards/`) so they don't
pollute the global fixture namespace. They depend on cross-cutting fixtures defined in
the parent `tests/conftest.py` (`device`, `resources_dir`, `density_map_1vme`), which
pytest resolves up the conftest hierarchy automatically.

`density_map_1vme` deliberately stays in the top-level conftest because
`tests/utils/test_density_utils.py` also consumes it; a fixture lives at the lowest
common ancestor of everything that uses it.

The reward families get their target data differently. The density fixtures read
committed files (a carved .ccp4 and its matching recentered cif). The SF fixtures
generate their (cif, mtz) pair per session via `generate_synthetic_sf.py`. The
diffuse_bragg fixtures expand 1vme's altlocs into a two-configuration ensemble and
generate Bragg and diffuse MTZs from it with the lunus-backed generator.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np
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
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction
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


# ---------------------------------------------------------------------------
# diffuse_bragg: a two-configuration ensemble and the targets generated from it
# ---------------------------------------------------------------------------

DIFFUSE_BRAGG_RESOLUTION = 1.8
DIFFUSE_BRAGG_TEST_FRACTION = 0.1


@dataclass(frozen=True)
class DiffuseBraggTargets:
    """MTZ targets generated from ``diffuse_bragg_ensemble``, and what they contain.

    Attributes
    ----------
    bragg
        ``<F>`` of the ensemble with a Phenix-convention ``R-free-flags`` column
        (1 = test), ``DIFFUSE_BRAGG_TEST_FRACTION`` of reflections flagged.
    diffuse
        ``<|F|²> − |<F>|²`` of the ensemble, column ``ID``.
    bragg_single
        ``|F|`` of the first configuration alone, no flag column. The target for the
        single-conformer contract tests, and the file without an integer column.
    cell, spacegroup
        Crystal frame all three were computed in.
    n_reflections, n_free
        Reflection count shared by the three files, and how many ``bragg`` flags free.
    """

    bragg: Path
    diffuse: Path
    bragg_single: Path
    cell: gemmi.UnitCell
    spacegroup: gemmi.SpaceGroup
    n_reflections: int
    n_free: int


@pytest.fixture(scope="session")
def diffuse_bragg_ensemble(resources_dir: Path) -> tuple[AtomArray, np.ndarray]:
    """1vme expanded into one configuration per altloc, waters stripped.

    Returns
    -------
    atom_array, coords
        The shared topology, and ``(n_configs, n_atoms, 3)`` float64 coordinates.
    """
    pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")
    from sampleworks.synthetic.generate_synthetic_sf import BatchRowForMTZ
    from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations

    source = resources_dir / "1vme" / "1vme_final.cif"
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")

    # Waters are stripped as guidance strips them; the model topology a reward is
    # prepared with carries none.
    configurations = load_configurations(
        source,
        BatchRowForMTZ(filename=source.name),
        "default",
        altlocs_as_models=True,
        strip_waters=True,
    )
    assert configurations.stack_depth() >= 2, "1vme should expand to at least two configurations"
    return configurations[0], np.asarray(configurations.coord, dtype=np.float64)


@pytest.fixture(scope="session")
def diffuse_bragg_targets(
    diffuse_bragg_ensemble: tuple[AtomArray, np.ndarray],
    resources_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> DiffuseBraggTargets:
    """Bragg and diffuse MTZs generated from the ensemble itself.

    Generating rather than checking in means the target and the thing being scored
    cannot drift apart, and recovery tests are a genuine round trip through the
    writers and the reader. Flags come from ``dataset_from_bragg_amplitudes``, the
    writer the CLI uses.
    """
    from biotite.structure import stack
    from sampleworks.synthetic.generate_synthetic_sf_lunus import (
        compute_ensemble_amplitudes,
        dataset_from_bragg_amplitudes,
        dataset_from_diffuse_intensities,
        save_mtz,
    )

    atom_array, coords = diffuse_bragg_ensemble
    meta = gemmi.read_structure(str(resources_dir / "1vme" / "1vme_final.cif"))
    cell, spacegroup = meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)
    device = torch.device("cpu")

    def configurations(xyz: np.ndarray):
        configs = stack([atom_array] * xyz.shape[0])
        configs.coord = xyz.astype(np.float32)
        return configs

    hkl, mean_f, diffuse = compute_ensemble_amplitudes(
        configurations(coords), cell, spacegroup, DIFFUSE_BRAGG_RESOLUTION, device
    )
    assert diffuse.max() > 0, "a two-state ensemble must have nonzero diffuse"
    hkl_single, f_single, _ = compute_ensemble_amplitudes(
        configurations(coords[:1]), cell, spacegroup, DIFFUSE_BRAGG_RESOLUTION, device
    )
    assert np.array_equal(hkl, hkl_single)

    out = tmp_path_factory.mktemp("diffuse_bragg_targets")
    bragg = dataset_from_bragg_amplitudes(
        hkl, mean_f, cell, spacegroup, test_fraction=DIFFUSE_BRAGG_TEST_FRACTION, seed=0
    )
    n_free = int(np.sum(bragg.column_with_label("R-free-flags").array == 1.0))
    save_mtz(bragg, out / "bragg.mtz", "structure factors")
    save_mtz(
        dataset_from_diffuse_intensities(hkl, diffuse, cell, spacegroup),
        out / "diffuse.mtz",
        "diffuse intensities",
    )
    save_mtz(
        dataset_from_bragg_amplitudes(hkl_single, f_single, cell, spacegroup, test_fraction=0.0),
        out / "bragg_single.mtz",
        "structure factors",
    )
    return DiffuseBraggTargets(
        bragg=out / "bragg.mtz",
        diffuse=out / "diffuse.mtz",
        bragg_single=out / "bragg_single.mtz",
        cell=cell,
        spacegroup=spacegroup,
        n_reflections=len(hkl),
        n_free=n_free,
    )


@pytest.fixture(scope="session")
def diffuse_bragg_rewards(
    diffuse_bragg_ensemble: tuple[AtomArray, np.ndarray],
    diffuse_bragg_targets: DiffuseBraggTargets,
) -> dict[float, "DiffuseBraggRewardFunction"]:
    """Prepared rewards on CPU, keyed by ``bragg_weight``: 1.0, 0.5 and 0.0.

    The pure terms are built with only the target they need, so constructing this
    fixture is itself the check that a zero weight does not require its target.
    """
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction
    from sampleworks.core.rewards.protocol import RewardInputs

    atom_array, _ = diffuse_bragg_ensemble
    targets = diffuse_bragg_targets
    rewards = {
        1.0: DiffuseBraggRewardFunction(bragg_target=targets.bragg, bragg_weight=1.0),
        0.5: DiffuseBraggRewardFunction(
            bragg_target=targets.bragg, diffuse_target=targets.diffuse, bragg_weight=0.5
        ),
        0.0: DiffuseBraggRewardFunction(diffuse_target=targets.diffuse, bragg_weight=0.0),
    }
    inputs = RewardInputs.from_atom_array(atom_array, ensemble_size=1)
    for reward in rewards.values():
        reward.prepare(inputs, device="cpu")
    return rewards


@pytest.fixture(scope="session")
def test_coordinates_1vme_diffuse_bragg(
    diffuse_bragg_ensemble: tuple[AtomArray, np.ndarray], device: torch.device
) -> tuple[Float[Tensor, "n_atoms 3"], AtomArray]:
    """The first configuration's coordinates and the topology, for ``_REWARD_BUNDLES``."""
    atom_array, coords = diffuse_bragg_ensemble
    return torch.as_tensor(coords[0], dtype=torch.float32, device=device), atom_array


@pytest.fixture(scope="session")
def reward_function_1vme_diffuse_bragg(
    diffuse_bragg_targets: DiffuseBraggTargets,
    diffuse_bragg_ensemble: tuple[AtomArray, np.ndarray],
    device: torch.device,
) -> "DiffuseBraggRewardFunction":
    """Pure-Bragg reward on the single-configuration target, prepared on ``device``.

    Bragg alone because the contract scores one conformer, for which the diffuse
    term is identically zero; the ensemble-valued behaviour is tested in
    ``test_diffuse_bragg_reward.py``. Prepared through ``prepare_reward_if_needed``,
    the hook the scalers use, so the hook's signature is exercised too.
    """
    from sampleworks.core.rewards.diffuse_bragg import DiffuseBraggRewardFunction
    from sampleworks.core.rewards.protocol import prepare_reward_if_needed, RewardInputs

    atom_array, _ = diffuse_bragg_ensemble
    reward = DiffuseBraggRewardFunction(
        bragg_target=diffuse_bragg_targets.bragg_single, bragg_weight=1.0
    )
    prepare_reward_if_needed(
        reward,
        RewardInputs.from_atom_array(atom_array, ensemble_size=1, device=device),
        device=device,
    )
    return reward
