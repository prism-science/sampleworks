"""Loss-specific pytest fixtures.

Fixtures here are scoped to the loss test suite (`tests/losses/`) so they don't
pollute the global fixture namespace. They depend on cross-cutting fixtures defined in
the parent `tests/conftest.py` (`device`, `resources_dir`, `density_map_1vme`), which
pytest resolves up the conftest hierarchy automatically.

`density_map_1vme` deliberately stays in the top-level conftest because
`tests/utils/test_density_utils.py` also consumes it; a fixture lives at the lowest
common ancestor of everything that uses it.

The loss families get their target data differently. The density fixtures read
committed files (a carved .ccp4 and its matching recentered cif). The SF fixtures
generate their (cif, mtz) pair per session via `generate_synthetic_sf_sfc.py`. The
diffuse_bragg fixtures run `generate_synthetic_sf_lunus.py`'s row processor on 1vme
three times: with its altlocs expanded into a two-configuration ensemble (Bragg,
diffuse and R-free flags), as a single model, and as the ensemble with bulk solvent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import pytest
import torch
from atomworks.io.parser import parse
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
        XMap,
    )
    from sampleworks.core.losses.diffuse_bragg import DiffuseBraggLossFunction
    from sampleworks.core.losses.real_space_density import RealSpaceLossFunction
    from sampleworks.core.losses.structure_factor import StructureFactorLossFunction


@pytest.fixture(scope="session")
def structure_1vme_density(resources_dir: Path) -> dict:
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Structure not found at {cif_path}")
    return parse(cif_path, ccd_mirror_path=None)


@pytest.fixture(scope="session")
def loss_function_1vme(density_map_1vme: "XMap", device: torch.device) -> "RealSpaceLossFunction":
    """Real-space density loss from the carved 1vme map, in X-ray (non-EM) mode."""
    from sampleworks.core.losses.real_space_density import (
        RealSpaceLossFunction,
        setup_scattering_params,
    )

    params = setup_scattering_params(em_mode=False, device=device)
    rf = RealSpaceLossFunction(density_map_1vme, params, torch.tensor([1], device=device))
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

    Mirrors the ``--structure`` branch of ``generate_synthetic_sf_sfc.main()``, so these
    fixtures exercise exactly what the CLI produces. ``--save-structure`` writes back the
    post-selection and occupancy modifed model, which is the structure actually used to
    generate the synthetic SF data.

    Specifically, we select for the chain A of 1vme with altloc occupancies forced to be
    0.5/0.5 and H + waters stripped, at 1.8 A resolution, with bulk solvent and default
    scales. The resulted mtz file carries both Fprotein/SIGFprotein/PHIFprotein and
    Ftotal/SIGFtotal/PHIFtotal.
    """
    from sampleworks.synthetic.generate_synthetic_sf_sfc import _process_single_row
    from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ

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
        bulk_solvent="combined",
        save_structure=True,
    )

    # Following the default file names in the generate_synthetic_sf_sfc.py script:
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
            f"generate_synthetic_sf_sfc wrote no {missing} to {output_dir}; see the "
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
    """Atomic coordinates and the atom array (asym_unit) which for the SF loss is the
    same atom array as the structure_1vme_sf.

    Exists for the ``_LOSS_BUNDLES`` mapping in ``test_loss_function_contract.py``,
    which resolves one ``(coords, atom_array)`` fixture per loss so every loss is
    exercised through the same shape. SF-specific tests that need only the atom array take
    ``structure_1vme_sf`` directly.
    """
    # No occupancy>0 filter here (unlike the density fixture) as SFC topology is fixed.
    coords = torch.from_numpy(structure_1vme_sf.coord).to(device=device, dtype=torch.float32)
    return coords, structure_1vme_sf


@pytest.fixture(scope="session")
def loss_function_1vme_sf(
    mtz_path_1vme: Path,
    structure_1vme_sf: AtomArray,
    device: torch.device,
) -> "StructureFactorLossFunction":
    """Structure factor loss (|Fprotein| only) prepared using the 1vme structure's topology."""
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.losses.structure_factor import StructureFactorLossFunction

    # normalize_amplitude=True scores normalized E-values (|Ec| vs sfc.Eo), which are
    # unit-variance per resolution shell, so the MSE can be tested on an absolute scale.
    loss_function = StructureFactorLossFunction(
        mtz_path_1vme,
        expcolumns=["Fprotein", "SIGFprotein"],
        normalize_amplitude=True,
    )
    loss_function.prepare(
        LossInputs.from_atom_array(structure_1vme_sf, ensemble_size=1, device=device),
        device=device,
    )
    return loss_function


# ---------------------------------------------------------------------------
# diffuse_bragg: 1vme through the lunus generator, as an ensemble and as one model
# ---------------------------------------------------------------------------

DIFFUSE_BRAGG_SOURCE = "1vme_final.cif"
DIFFUSE_BRAGG_RESOLUTION = 1.8
DIFFUSE_BRAGG_TEST_FRACTION = 0.1


@dataclass(frozen=True)
class DiffuseBraggTargets:
    """MTZ targets written by the lunus generator, and what they contain.

    Attributes
    ----------
    bragg
        ``<F>`` of the two-configuration ensemble with a Phenix-convention
        ``R-free-flags`` column (1 = test), ``DIFFUSE_BRAGG_TEST_FRACTION`` flagged.
    diffuse
        ``<|F|²> − |<F>|²`` of the ensemble, column ``ID``.
    bragg_single
        ``|F|`` of 1vme loaded as one model (altlocs as partial-occupancy atoms), no
        flag column. The target for the single-conformer contract bundle, and the
        Bragg file without an integer column.
    bragg_solvent, diffuse_solvent
        The ensemble targets again, with a per-configuration bulk-solvent
        contribution at the generator's calibrated cutoff.
    cell, spacegroup
        Crystal frame all of them were computed in.
    n_reflections, n_free
        Reflection count shared by the files, and how many ``bragg`` flags free.
    """

    bragg: Path
    diffuse: Path
    bragg_single: Path
    bragg_solvent: Path
    diffuse_solvent: Path
    cell: gemmi.UnitCell
    spacegroup: gemmi.SpaceGroup
    n_reflections: int
    n_free: int


def _generate_lunus_targets(
    resources_dir: Path, output_dir: Path, device: torch.device, **options
) -> dict:
    """Run the lunus generator's row processor on 1vme and return its record.

    Mirrors the ``--structure`` branch of ``generate_synthetic_sf_lunus.main()``, so
    the fixtures exercise exactly what the CLI writes. The processor logs and
    returns on failure instead of raising, so the record's status is checked here.
    """
    from sampleworks.synthetic.generate_synthetic_sf_lunus import _process_single_row
    from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ

    record = _process_single_row(
        row=BatchRowForMTZ(filename=DIFFUSE_BRAGG_SOURCE),
        base_dir=resources_dir / "1vme",
        output_dir=output_dir,
        resolution=DIFFUSE_BRAGG_RESOLUTION,
        occupancy_mode="default",
        seed=0,
        device=device,
        strip_waters=True,
        **options,
    )
    if record["status"] != "success":
        raise RuntimeError(
            f"generate_synthetic_sf_lunus failed at {record.get('failure_stage')}: "
            f"{record.get('error')}"
        )
    return record


def _lunus_configurations(resources_dir: Path, altlocs_as_models: bool) -> AtomArrayStack:
    """1vme as the lunus generator loads it, waters stripped (guidance strips them too)."""
    pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")
    from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations
    from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ

    source = resources_dir / "1vme" / DIFFUSE_BRAGG_SOURCE
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")
    return load_configurations(
        source,
        BatchRowForMTZ(filename=DIFFUSE_BRAGG_SOURCE),
        "default",
        altlocs_as_models=altlocs_as_models,
        strip_waters=True,
    )


@pytest.fixture(scope="session")
def diffuse_bragg_ensemble(resources_dir: Path) -> tuple[AtomArray, np.ndarray]:
    """1vme expanded into one configuration per altloc.

    Returns
    -------
    tuple[AtomArray, np.ndarray]
        The shared topology, and float64 coordinates ``[n_configs, n_atoms, 3]``.
    """
    configurations = _lunus_configurations(resources_dir, altlocs_as_models=True)
    assert configurations.stack_depth() >= 2, "1vme should expand to at least two configurations"
    return configurations[0], np.asarray(configurations.coord, dtype=np.float64)


@pytest.fixture(scope="session")
def diffuse_bragg_single_model(resources_dir: Path) -> AtomArray:
    """1vme as one model, altlocs kept as partial-occupancy atoms."""
    return _lunus_configurations(resources_dir, altlocs_as_models=False)[0]


@pytest.fixture(scope="session")
def diffuse_bragg_targets(
    resources_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> DiffuseBraggTargets:
    """Bragg and diffuse MTZs written by the lunus generator from 1vme.

    Generating rather than checking in means the target and the thing being scored
    cannot drift apart, and recovery tests are a genuine round trip through the
    generator, the writers and the reader.
    """
    device = torch.device("cpu")
    out = tmp_path_factory.mktemp("diffuse_bragg_targets")
    ensemble = _generate_lunus_targets(
        resources_dir,
        out / "ensemble",
        device,
        test_fraction=DIFFUSE_BRAGG_TEST_FRACTION,
        write_diffuse=True,
        altlocs_as_models=True,
    )
    single = _generate_lunus_targets(resources_dir, out / "single", device, test_fraction=0.0)
    solvent = _generate_lunus_targets(
        resources_dir,
        out / "solvent",
        device,
        test_fraction=0.0,
        write_diffuse=True,
        altlocs_as_models=True,
        bulk_solvent=True,
    )

    bragg = gemmi.read_mtz_file(ensemble["output_path"])
    # Same cell and resolution every time, so one reflection list serves all five files.
    for record in (ensemble, single, solvent):
        for key in ("output_path", "diffuse_output_path"):
            if key in record:
                assert gemmi.read_mtz_file(record[key]).nreflections == bragg.nreflections
    return DiffuseBraggTargets(
        bragg=Path(ensemble["output_path"]),
        diffuse=Path(ensemble["diffuse_output_path"]),
        bragg_single=Path(single["output_path"]),
        bragg_solvent=Path(solvent["output_path"]),
        diffuse_solvent=Path(solvent["diffuse_output_path"]),
        cell=bragg.cell,
        spacegroup=bragg.spacegroup,
        n_reflections=bragg.nreflections,
        n_free=int(np.sum(bragg.column_with_label("R-free-flags").array == 1.0)),
    )


@pytest.fixture(scope="session")
def diffuse_bragg_losses(
    diffuse_bragg_ensemble: tuple[AtomArray, np.ndarray],
    diffuse_bragg_targets: DiffuseBraggTargets,
) -> dict[float, "DiffuseBraggLossFunction"]:
    """Prepared losses on CPU, keyed by ``bragg_weight``: 1.0, 0.5 and 0.0.

    The pure terms are built with only the target they need, so constructing this
    fixture is itself the check that a zero weight does not require its target.
    """
    from sampleworks.core.losses.diffuse_bragg import DiffuseBraggLossFunction
    from sampleworks.core.losses.protocol import LossInputs

    atom_array, _ = diffuse_bragg_ensemble
    targets = diffuse_bragg_targets
    losses = {
        1.0: DiffuseBraggLossFunction(bragg_target=targets.bragg, bragg_weight=1.0),
        0.5: DiffuseBraggLossFunction(
            bragg_target=targets.bragg, diffuse_target=targets.diffuse, bragg_weight=0.5
        ),
        0.0: DiffuseBraggLossFunction(diffuse_target=targets.diffuse, bragg_weight=0.0),
    }
    inputs = LossInputs.from_atom_array(atom_array, ensemble_size=1)
    for loss in losses.values():
        loss.prepare(inputs, device="cpu")
    return losses


@pytest.fixture(scope="session")
def test_coordinates_1vme_diffuse_bragg(
    diffuse_bragg_single_model: AtomArray, device: torch.device
) -> tuple[Float[Tensor, "n_atoms 3"], AtomArray]:
    """The single-model coordinates and topology, for ``_LOSS_BUNDLES``."""
    coords = torch.as_tensor(diffuse_bragg_single_model.coord, dtype=torch.float32, device=device)
    return coords, diffuse_bragg_single_model


@pytest.fixture(scope="session")
def loss_function_1vme_diffuse_bragg(
    diffuse_bragg_targets: DiffuseBraggTargets,
    diffuse_bragg_single_model: AtomArray,
    device: torch.device,
) -> "DiffuseBraggLossFunction":
    """Pure-Bragg loss on the single-model target, prepared on ``device``.

    Bragg alone because the contract scores one conformer, for which the diffuse
    term is identically zero; the ensemble-valued behaviour is tested in
    ``test_diffuse_bragg_loss.py``. Prepared through ``prepare_loss_if_needed``,
    the hook the scalers use, so the hook's signature is exercised too.
    """
    from sampleworks.core.losses.diffuse_bragg import DiffuseBraggLossFunction
    from sampleworks.core.losses.protocol import LossInputs, prepare_loss_if_needed

    loss = DiffuseBraggLossFunction(
        bragg_target=diffuse_bragg_targets.bragg_single, bragg_weight=1.0
    )
    prepare_loss_if_needed(
        loss,
        LossInputs.from_atom_array(diffuse_bragg_single_model, ensemble_size=1, device=device),
        device=device,
    )
    return loss
