"""Shared pytest fixtures for sampleworks tests."""

import importlib
import inspect
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from site import getsitepackages
from typing import Any, TYPE_CHECKING

import numpy as np
import pytest
import torch


_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atomworks.io.parser import parse, parse_atom_array
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, stack
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.protocol import StepParams
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.guidance_constants import (
    Rewards,
    StepScalers,
    StructurePredictor,
    TrajectorySamplers,
    TrajectoryScalers,
)
from sampleworks.utils.imports import (
    BOLTZ_AVAILABLE,
    PROTENIX_AVAILABLE,
    PROTPARDELLE_AVAILABLE,
    RF3_AVAILABLE,
)
from sampleworks.utils.structure_utils import SampleworksProcessedStructure
from sampleworks.utils.torch_utils import try_gpu

from tests.mocks import MockFlowModelWrapper, MockStepScaler
from tests.mocks.rewards import MockGradientRewardFunction


if TYPE_CHECKING:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
    )
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
    from sampleworks.models.protpardelle.wrapper import ProtpardelleWrapper
    from sampleworks.models.rf3.wrapper import RF3Wrapper

if BOLTZ_AVAILABLE:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
    )

if PROTENIX_AVAILABLE:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper

if PROTPARDELLE_AVAILABLE:
    from sampleworks.models.protpardelle.wrapper import ProtpardelleWrapper

if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import RF3Wrapper


def _import_from_path(path: str) -> Any:
    """Dynamically import an object from a fully qualified path."""
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


# ============================================================================
# Component creation from registry
# ============================================================================


@dataclass(frozen=True)
class ComponentInfo:
    """Metadata about a component for test parametrization.

    Parameters
    ----------
    name
        Human-readable name for test IDs.
    module_path
        Path to the class.
    requires_checkpoint
        Whether this component needs model checkpoints.
    is_trajectory_sampler
        For samplers, whether it implements TrajectorySampler.
    requires_reward
        For scalers, whether it requires a reward function.
    default_kwargs
        Default kwargs for instantiation in tests.
    config_class_path
        For components (e.g. AF3EDMSampler) whose constructor takes a config object,
        the path to the config class. When set, default_kwargs are passed to the
        config constructor, and the resulting config is then passed to the component
        constructor.
    annotate_fn_path
        For wrappers, fully qualified path to the annotate function.
    conditioning_type_path
        For wrappers, fully qualified path to the conditioning type.
    requires_out_dir
        For wrappers, whether the annotate function requires out_dir.
    """

    name: str
    module_path: str
    requires_checkpoint: bool = False
    is_trajectory_sampler: bool = False
    requires_reward: bool = False
    default_kwargs: tuple[tuple[str, Any], ...] = ()
    config_class_path: str = ""
    annotate_fn_path: str = ""
    conditioning_type_path: str = ""
    requires_out_dir: bool = True


MODEL_WRAPPER_REGISTRY: dict[StructurePredictor, ComponentInfo] = {
    StructurePredictor.BOLTZ_1: ComponentInfo(
        name="boltz1",
        module_path="sampleworks.models.boltz.wrapper.Boltz1Wrapper",
        requires_checkpoint=True,
        annotate_fn_path="sampleworks.models.boltz.wrapper.process_structure_for_boltz",
        conditioning_type_path="sampleworks.models.boltz.wrapper.BoltzConditioning",
        requires_out_dir=True,
    ),
    StructurePredictor.BOLTZ_2: ComponentInfo(
        name="boltz2",
        module_path="sampleworks.models.boltz.wrapper.Boltz2Wrapper",
        requires_checkpoint=True,
        annotate_fn_path="sampleworks.models.boltz.wrapper.process_structure_for_boltz",
        conditioning_type_path="sampleworks.models.boltz.wrapper.BoltzConditioning",
        requires_out_dir=True,
    ),
    StructurePredictor.PROTENIX: ComponentInfo(
        name="protenix",
        module_path="sampleworks.models.protenix.wrapper.ProtenixWrapper",
        requires_checkpoint=True,
        annotate_fn_path="sampleworks.models.protenix.wrapper.annotate_structure_for_protenix",
        conditioning_type_path="sampleworks.models.protenix.wrapper.ProtenixConditioning",
        requires_out_dir=True,
    ),
    StructurePredictor.RF3: ComponentInfo(
        name="rf3",
        module_path="sampleworks.models.rf3.wrapper.RF3Wrapper",
        requires_checkpoint=True,
        annotate_fn_path="sampleworks.models.rf3.wrapper.annotate_structure_for_rf3",
        conditioning_type_path="sampleworks.models.rf3.wrapper.RF3Conditioning",
        requires_out_dir=False,
    ),
    StructurePredictor.PROTPARDELLE: ComponentInfo(
        name="protpardelle",
        module_path="sampleworks.models.protpardelle.wrapper.ProtpardelleWrapper",
        requires_checkpoint=True,
        annotate_fn_path="sampleworks.models.protpardelle.wrapper.annotate_structure_for_protpardelle",
        conditioning_type_path="sampleworks.models.protpardelle.wrapper.ProtpardelleConditioning",
        requires_out_dir=False,
    ),
}


SAMPLER_REGISTRY: dict[TrajectorySamplers, ComponentInfo] = {
    TrajectorySamplers.AF3EDM: ComponentInfo(
        name="af3edm",
        module_path="sampleworks.core.samplers.edm.AF3EDMSampler",
        config_class_path="sampleworks.core.samplers.edm.EDMSamplerConfig",
        is_trajectory_sampler=True,
    ),
}


STEP_SCALER_REGISTRY: dict[StepScalers, ComponentInfo] = {
    StepScalers.NO_SCALING: ComponentInfo(
        name="no_scaling",
        module_path="sampleworks.core.scalers.step_scalers.NoScalingScaler",
        requires_reward=False,
    ),
    StepScalers.DATA_SPACE_DPS: ComponentInfo(
        name="data_space_dps",
        module_path="sampleworks.core.scalers.step_scalers.DataSpaceDPSScaler",
        requires_reward=True,
        default_kwargs=(("step_size", 0.1),),
    ),
    StepScalers.NOISE_SPACE_DPS: ComponentInfo(
        name="noise_space_dps",
        module_path="sampleworks.core.scalers.step_scalers.NoiseSpaceDPSScaler",
        requires_reward=True,
        default_kwargs=(("step_size", 0.1),),
    ),
}


TRAJECTORY_SCALER_REGISTRY: dict[TrajectoryScalers, ComponentInfo] = {
    TrajectoryScalers.PURE_GUIDANCE: ComponentInfo(
        name="pure_guidance",
        module_path="sampleworks.core.scalers.pure_guidance.PureGuidance",
        default_kwargs=(("ensemble_size", 3), ("num_steps", 5)),
    ),
    TrajectoryScalers.FK_STEERING: ComponentInfo(
        name="fk_steering",
        module_path="sampleworks.core.scalers.fk_steering.FKSteering",
        default_kwargs=(("ensemble_size", 3), ("num_steps", 5)),
    ),
}


REWARD_REGISTRY: dict[Rewards, ComponentInfo] = {
    Rewards.REAL_SPACE_DENSITY: ComponentInfo(
        name="real_space_density",
        module_path="sampleworks.core.rewards.real_space_density.RealSpaceRewardFunction",
        requires_checkpoint=True,
    ),
}


def create_component_from_info(
    info: ComponentInfo,
    device: torch.device | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create a component instance from ComponentInfo.

    Parameters
    ----------
    info
        ComponentInfo from a registry.
    device
        Optional device to pass if the component accepts it.
    **extra_kwargs
        Additional kwargs to override defaults.

    Returns
    -------
    Any
        The instantiated component.
    """
    cls = _import_from_path(info.module_path)
    kwargs = dict(info.default_kwargs)
    kwargs.update(extra_kwargs)
    sig = inspect.signature(cls)
    if device is not None and "device" in sig.parameters:
        kwargs["device"] = device
    return cls(**kwargs)


def create_sampler_from_type(
    sampler_type: TrajectorySamplers,
    device: torch.device | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create sampler from TrajectorySamplers enum."""
    info = SAMPLER_REGISTRY[sampler_type]
    if info.config_class_path:
        config_cls = _import_from_path(info.config_class_path)
        config_kwargs = dict(info.default_kwargs)
        config_kwargs.update(extra_kwargs)
        if device is not None:
            config_kwargs["device"] = device
        config = config_cls(**config_kwargs)
        cls = _import_from_path(info.module_path)
        return cls(config)
    return create_component_from_info(info, device=device, **extra_kwargs)


def create_step_scaler_from_type(
    scaler_type: StepScalers,
    **extra_kwargs: Any,
) -> Any:
    """Create step scaler from StepScalers enum."""
    info = STEP_SCALER_REGISTRY[scaler_type]
    return create_component_from_info(info, **extra_kwargs)


def create_trajectory_scaler_from_type(
    scaler_type: TrajectoryScalers,
    **extra_kwargs: Any,
) -> Any:
    """Create trajectory scaler from TrajectoryScalers enum."""
    info = TRAJECTORY_SCALER_REGISTRY[scaler_type]
    return create_component_from_info(info, **extra_kwargs)


def create_reward_from_type(
    reward_type: Rewards,
    **extra_kwargs: Any,
) -> Any:
    """Create reward from Rewards enum."""
    info = REWARD_REGISTRY[reward_type]
    return create_component_from_info(info, **extra_kwargs)


# ============================================================================
# Mock wrapper configuration (not part of main registry)
# ============================================================================

MOCK_WRAPPER_INFO = ComponentInfo(
    name="mock",
    module_path="tests.mocks.model_wrappers.MockFlowModelWrapper",
    requires_checkpoint=False,
    annotate_fn_path="",
    conditioning_type_path="",
    requires_out_dir=False,
)


# ============================================================================
# Registry helpers for test parametrization
# ============================================================================

STRUCTURES: list[str] = [
    "structure_1vme",
    "structure_6b8x",
    "structure_2yl0",
    "structure_5sop",
    "structure_6ni6",
    "structure_9bn8",
]


def get_all_model_wrappers() -> list[StructurePredictor]:
    """Get all model wrapper types from the registry."""
    return list(MODEL_WRAPPER_REGISTRY.keys())


def get_all_trajectory_samplers() -> list[TrajectorySamplers]:
    """Get all trajectory sampler types from the registry."""
    return list(SAMPLER_REGISTRY.keys())


def get_all_step_scalers() -> list[StepScalers]:
    """Get all step scaler types from the registry."""
    return list(STEP_SCALER_REGISTRY.keys())


def get_all_trajectory_scalers() -> list[TrajectoryScalers]:
    """Get all trajectory scaler types from the registry."""
    return list(TRAJECTORY_SCALER_REGISTRY.keys())


def get_all_rewards() -> list[Rewards]:
    """Get all reward types from the registry."""
    return list(REWARD_REGISTRY.keys())


def get_slow_wrappers() -> list[StructurePredictor]:
    """Get all model wrappers that require checkpoints (for slow tests).

    All model wrappers in MODEL_WRAPPER_REGISTRY require checkpoints,
    so this returns the same as get_all_model_wrappers().
    """
    return [
        wrapper_type
        for wrapper_type, info in MODEL_WRAPPER_REGISTRY.items()
        if info.requires_checkpoint
    ]


def get_fixture_name_for_wrapper(info: ComponentInfo) -> str:
    """Get the pytest fixture name for a wrapper info."""
    return f"{info.name}_wrapper"


def create_model_wrapper_from_type(
    predictor: StructurePredictor,
    device: torch.device | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create model wrapper from StructurePredictor enum."""
    info = MODEL_WRAPPER_REGISTRY[predictor]
    return create_component_from_info(info, device=device, **extra_kwargs)


def get_annotate_fn(wrapper: ComponentInfo) -> Callable[..., dict]:
    """Get the annotate function for a wrapper configuration."""
    if not wrapper.annotate_fn_path:
        raise ValueError(f"Wrapper {wrapper.name} does not have an annotate function")
    return _import_from_path(wrapper.annotate_fn_path)


def get_conditioning_type(wrapper: ComponentInfo) -> type:
    """Get the conditioning type for a wrapper configuration."""
    if not wrapper.conditioning_type_path:
        raise ValueError(f"Wrapper {wrapper.name} does not have a conditioning type")
    return _import_from_path(wrapper.conditioning_type_path)


def annotate_structure_for_wrapper(
    wrapper: ComponentInfo,
    structure: dict,
    temp_output_dir: Path | None = None,
    **kwargs: Any,
) -> dict:
    """Call the appropriate annotate function for a wrapper configuration.

    Parameters
    ----------
    wrapper
        The ComponentInfo for the wrapper.
    structure
        The structure dictionary to annotate.
    temp_output_dir
        Temporary output directory (required if wrapper.requires_out_dir is True).
    **kwargs
        Additional keyword arguments passed to the annotate function.

    Returns
    -------
    dict
        The annotated structure.
    """
    annotate_fn = get_annotate_fn(wrapper)
    if wrapper.requires_out_dir:
        return annotate_fn(structure, out_dir=temp_output_dir, **kwargs)
    return annotate_fn(structure, **kwargs)


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    return Path(__file__).parent / "resources"


def parse_and_remove_hydrogens(path: Path | str) -> dict:
    """
    Parse an mmCIF structure file with atomworks.io.parser.parse with settings to remove hydrogens.
    Parameters
    ----------
    path : Path | str
        path to the mmCIF structure file.

    Returns
    _______
        The parsed structure as a dictionary. See atomworks.io.parser.parse for details.
    """
    return parse(Path(path), ccd_mirror_path=None, hydrogen_policy="remove")


@pytest.fixture(scope="session")
def structure_1vme(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "1vme" / "1vme_final.cif")


@pytest.fixture(scope="session")
def atom_array_1vme_with_missing_atoms(structure_1vme) -> AtomArray:
    """1VME atom array after parse_atom_array with add_missing_atoms=True.

    This reproduces what RF3's InferenceInput.from_atom_array does internally.
    """
    parsed = parse_atom_array(
        structure_1vme["asym_unit"],
        add_missing_atoms=True,
        hydrogen_policy="keep",
    )
    aa = parsed["asym_unit"]
    if isinstance(aa, AtomArrayStack):
        aa = aa[0]
    return aa


@pytest.fixture(scope="session")
def structure_6b8x(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "6b8x" / "6b8x_final.pdb")


@pytest.fixture(scope="session")
def structure_2yl0(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "2YL0" / "2YL0_single_001.pdb")


@pytest.fixture(scope="session")
def structure_2yl0_density(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "2YL0" / "2YL0_single_001_density_input.cif")


@pytest.fixture(scope="session")
def structure_5sop(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "5SOP" / "5SOP_single_001.pdb")


@pytest.fixture(scope="session")
def structure_5sop_density(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "5SOP" / "5SOP_single_001_density_input.cif")


@pytest.fixture(scope="session")
def structure_6ni6(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "6NI6" / "6NI6_single_001.pdb")


@pytest.fixture(scope="session")
def structure_6ni6_density(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "6NI6" / "6NI6_single_001_density_input.cif")


@pytest.fixture(scope="session")
def structure_9bn8(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "9BN8" / "9BN8_single_001.pdb")


@pytest.fixture(scope="session")
def structure_9bn8_density(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "9BN8" / "9BN8_single_001_density_input.cif")


@pytest.fixture(scope="session")
def structure_5i09_density(resources_dir: Path) -> dict:
    return parse_and_remove_hydrogens(resources_dir / "5I09" / "5I09_single_001_density_input.cif")


@pytest.fixture(scope="session")
def structure_6b8x_with_altlocs(resources_dir: Path) -> AtomArray | AtomArrayStack:
    return load_any(
        resources_dir / "6b8x" / "6b8x_final.pdb", altloc="all", extra_fields=["occupancy"]
    )


@pytest.fixture(scope="session", params=["1vme_final.cif", "6b8x_final.pdb"], ids=["cif", "pdb"])
def test_structure(request, resources_dir: Path) -> dict:
    # this requires the 1st 4 characters of the filename to match the folder name,
    # so PDB IDs need to be matching case
    return parse_and_remove_hydrogens(resources_dir / request.param[:4] / request.param)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return try_gpu()


# ============================================================================
# Parametrized component fixtures for protocol tests
# ============================================================================


@pytest.fixture(params=get_all_model_wrappers(), ids=lambda w: w.value)
def model_wrapper_type(request: pytest.FixtureRequest) -> StructurePredictor:
    """Provides model wrapper type for all implementations."""
    return request.param


@pytest.fixture(params=get_all_trajectory_samplers(), ids=lambda s: s.value)
def trajectory_sampler_type(request: pytest.FixtureRequest) -> TrajectorySamplers:
    """Provides trajectory sampler type for all implementations."""
    return request.param


@pytest.fixture(params=get_all_step_scalers(), ids=lambda s: s.value)
def step_scaler_type(request: pytest.FixtureRequest) -> StepScalers:
    """Provides step scaler type for all implementations."""
    return request.param


@pytest.fixture(params=get_all_trajectory_scalers(), ids=lambda s: s.value)
def trajectory_scaler_type(request: pytest.FixtureRequest) -> TrajectoryScalers:
    """Provides trajectory scaler type for all implementations."""
    return request.param


def get_wrapper_info(wrapper_type: StructurePredictor) -> ComponentInfo:
    """Get ComponentInfo for a wrapper type."""
    return MODEL_WRAPPER_REGISTRY[wrapper_type]


def get_fixture_name_for_wrapper_type(wrapper_type: StructurePredictor) -> str:
    """Get the pytest fixture name for a wrapper type."""
    info = MODEL_WRAPPER_REGISTRY[wrapper_type]
    return f"{info.name}_wrapper"


def annotate_structure_for_wrapper_type(
    wrapper_type: StructurePredictor,
    structure: dict,
    temp_output_dir: Path | None = None,
    **kwargs: Any,
) -> dict:
    """Annotate a structure for a specific wrapper type.

    Parameters
    ----------
    wrapper_type
        The StructurePredictor enum value.
    structure
        The structure dictionary to annotate.
    temp_output_dir
        Temporary output directory.
    **kwargs
        Additional keyword arguments passed to the annotate function.

    Returns
    -------
    dict
        The annotated structure.
    """
    info = MODEL_WRAPPER_REGISTRY[wrapper_type]
    return annotate_structure_for_wrapper(info, structure, temp_output_dir, **kwargs)


@pytest.fixture(scope="session")
def boltz1_checkpoint_path() -> Path:
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    path = Path("~/.boltz/boltz1_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz1 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def boltz2_checkpoint_path() -> Path:
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    path = Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz2 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def boltz1_wrapper(boltz1_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    return Boltz1Wrapper(
        checkpoint_path=boltz1_checkpoint_path,
        use_msa_manager=True,
        device=device,
    )


@pytest.fixture(scope="session")
def boltz2_wrapper(boltz2_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    return Boltz2Wrapper(
        checkpoint_path=boltz2_checkpoint_path,
        use_msa_manager=True,
        device=device,
    )


@pytest.fixture(scope="session")
def protenix_checkpoint_path() -> Path:
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    # Protenix would auto-download the checkpoint on first use if missing. Skip instead of
    # reaching out to the network so non-GPU/offline runs (e.g. CI) don't hang or time out
    # constructing the wrapper (see #216). Provision the checkpoint to run these tests.
    path = Path(getsitepackages()[0]) / "release_data/checkpoint/protenix_base_default_v0.5.0.pt"
    if not path.exists():
        pytest.skip(f"Protenix checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def protenix_wrapper(protenix_checkpoint_path: Path, device: torch.device):
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    return ProtenixWrapper(
        checkpoint_path=protenix_checkpoint_path,
        device=device,
    )


@pytest.fixture(scope="session")
def rf3_checkpoint_path() -> Path:
    if not RF3_AVAILABLE:
        pytest.skip("RF3 dependencies not installed in this environment")
    path = Path("~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"RF3 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def rf3_wrapper(rf3_checkpoint_path: Path):  # will run on Fabric device
    if not RF3_AVAILABLE:
        pytest.skip("RF3 dependencies not installed in this environment")
    return RF3Wrapper(
        checkpoint_path=rf3_checkpoint_path,
    )


@pytest.fixture(scope="session")
def protpardelle_checkpoint_path() -> Path:
    if not PROTPARDELLE_AVAILABLE:
        pytest.skip("Protpardelle dependencies not installed in this environment")
    import os

    # where we will eventually install the checkpoint on our containers.
    builtin_path = Path(getsitepackages()[0]) / "release_data" / "protpardelle" / "model_params"

    # if the user specifies a path to the model_params directory, use that instead.
    path = Path(os.environ.get("PROTPARDELLE_MODEL_PARAMS", builtin_path))
    path = path / "weights/cc89_epoch415.pth"
    return path


@pytest.fixture(scope="session")
def protpardelle_wrapper(protpardelle_checkpoint_path: Path, device: torch.device):
    if not PROTPARDELLE_AVAILABLE:
        pytest.skip("Protpardelle dependencies not installed in this environment")

    config_path = files("sampleworks.data").joinpath("cc89_epoch415.yaml")
    return ProtpardelleWrapper(
        config_path=str(Path(config_path).expanduser().resolve()),
        checkpoint_path=protpardelle_checkpoint_path,
        device=device,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    output_dir = tmp_path / "boltz_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir


@pytest.fixture(scope="session")
def density_map_1vme(resources_dir: Path):
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
        XMap,
    )

    map_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    if not map_path.exists():
        pytest.skip(f"Density map not found at {map_path}")
    return XMap.fromfile(str(map_path), resolution=1.8)


@pytest.fixture(scope="module")
def atom_array_with_nan_coords():
    """AtomArray with some NaN coordinates."""

    atom_array = AtomArray(5)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, np.inf, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture(scope="module")
def simple_atom_array_stack():
    """AtomArrayStack with 2 models."""

    arrays = []
    for i in range(2):
        atom_array = AtomArray(3)
        base_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_array.coord = base_coords + i * 0.1
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0]))
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)

    return atom_array_stack


@pytest.fixture(scope="module")
def atom_array_stack_uniform_occ():
    """AtomArrayStack with 2 models and occupancy 0.5"""

    arrays = []
    for i in range(2):
        atom_array = AtomArray(3)
        base_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_array.coord = base_coords + i * 0.1
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.5]))
        arrays.append(atom_array)

    return stack(arrays)


@pytest.fixture(scope="module")
def atom_array_stack_with_nan_coords():
    """AtomArrayStack with 2 models where one model has a NaN coordinate."""

    a1 = AtomArray(3)
    a1.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    a1.set_annotation("chain_id", np.array(["A"] * 3))
    a1.set_annotation("res_id", np.array([1, 2, 3]))
    a1.set_annotation("element", np.array(["C", "C", "C"]))
    a1.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
    a1.set_annotation("occupancy", np.array([0.5, 0.5, 0.5]))

    a2 = a1.copy()
    a2.coord = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [2.0, 0.0, 0.0]])

    return stack([a1, a2])


@pytest.fixture(scope="module")
def basic_atom_array_altloc():
    """AtomArray with mixed altloc_ids."""

    atom_array = AtomArray(5)
    coord = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A", "A", "A", "A", "A"]))
    atom_array.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_full_occupancy():
    """AtomArray with some atoms having full occupancy."""

    atom_array = AtomArray(8)
    atom_array.coord = np.random.rand(8, 3)
    atom_array.set_annotation("chain_id", np.array(["A"] * 8))
    atom_array.set_annotation("res_id", [1, 1, 2, 3, 4, 5, 5, 5])
    atom_array.set_annotation("res_name", np.array(["ALA"] * 8))
    atom_array.set_annotation("atom_name", np.array(["CA"] * 8))
    atom_array.set_annotation("element", np.array(["C"] * 8))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "A", "A", "A", "B", "C"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.4, 1.0, 0.7, 0.2, 0.1]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack_altloc():
    """AtomArrayStack with 3 models."""

    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array_stack.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))
    atom_array_stack.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array_stack.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))

    return atom_array_stack


@pytest.fixture(scope="module")
def atom_array_missing_altloc_id():
    """AtomArray without altloc_id annotation."""

    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("occupancy", np.ones(5))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_missing_occupancy():
    """AtomArray without occupancy annotation."""

    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("altloc_id", np.array(["A"] * 5))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_partial_overlap():
    """Two AtomArrays with partial overlap."""

    array1 = AtomArray(5)
    array1.coord = np.random.rand(5, 3)
    array1.set_annotation("chain_id", np.array(["A"] * 5))
    array1.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    array1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    array1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array1.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    array2 = AtomArray(5)
    array2.coord = np.random.rand(5, 3)
    array2.set_annotation("chain_id", np.array(["A"] * 5))
    array2.set_annotation("res_id", np.array([3, 4, 5, 6, 7]))
    array2.set_annotation("res_name", np.array(["VAL", "LEU", "SER", "THR", "TYR"]))
    array2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array2.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    return array1, array2


@pytest.fixture(scope="module")
def atom_array_stacks_partial_overlap():
    """Two AtomArrayStacks with partial overlap."""

    arrays1 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays1.append(array)
    stack1 = stack(arrays1)
    stack1.set_annotation("chain_id", np.array(["A"] * 4))
    stack1.set_annotation("res_id", np.array([1, 2, 3, 4]))
    stack1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU"]))
    stack1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack1.set_annotation("element", np.array(["C", "C", "C", "C"]))

    arrays2 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays2.append(array)
    stack2 = stack(arrays2)
    stack2.set_annotation("chain_id", np.array(["A"] * 4))
    stack2.set_annotation("res_id", np.array([2, 3, 4, 5]))
    stack2.set_annotation("res_name", np.array(["GLY", "VAL", "LEU", "SER"]))
    stack2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack2.set_annotation("element", np.array(["C", "C", "C", "C"]))

    return stack1, stack2


@pytest.fixture(scope="module")
def basic_atom_array_multichain():
    """AtomArray with chain_id, res_id, atom_name annotations."""

    atom_array = AtomArray(10)
    coord = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation(
        "chain_id", np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    )
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))
    atom_array.set_annotation(
        "res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER", "ALA", "GLY", "VAL", "LEU", "SER"])
    )
    atom_array.set_annotation(
        "atom_name", np.array(["CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA"])
    )
    atom_array.set_annotation(
        "element", np.array(["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"])
    )
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack_simple():
    """AtomArrayStack with 3 models."""

    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.arange(1, 6))
    atom_array_stack.set_annotation("res_name", np.array(["ALA"] * 5))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))

    return atom_array_stack


# ============================================================================
# Mock wrapper and sampler fixtures for fast testing
# ============================================================================


@pytest.fixture
def mock_wrapper(device: torch.device) -> MockFlowModelWrapper:
    """MockFlowModelWrapper with default settings."""
    return MockFlowModelWrapper(num_atoms=50, device=device)


@pytest.fixture
def mock_structure() -> dict:
    """Mock structure dict for testing without real PDB files."""
    n_atoms = 50
    atom_array = AtomArray(n_atoms)
    atom_array.coord = np.random.randn(n_atoms, 3).astype(np.float32)
    atom_array.element = np.array(["C"] * n_atoms)
    atom_array.atom_name = np.array(["CA"] * n_atoms)
    atom_array.res_name = np.array(["ALA"] * n_atoms)
    atom_array.res_id = np.arange(1, n_atoms + 1)
    atom_array.chain_id = np.array(["A"] * n_atoms)
    atom_array.set_annotation("occupancy", np.ones(n_atoms))
    atom_array.set_annotation("b_factor", np.ones(n_atoms) * 20.0)
    return {"asym_unit": atom_array, "metadata": {"id": "mock"}}


@pytest.fixture
def converging_mock_wrapper(device: torch.device) -> MockFlowModelWrapper:
    """MockFlowModelWrapper configured for convergence testing."""
    target = torch.randn(1, 50, 3, device=device)
    return MockFlowModelWrapper(num_atoms=50, device=device, target=target)


@pytest.fixture
def edm_sampler(device: torch.device) -> AF3EDMSampler:
    """AF3EDMSampler configured for testing."""
    config = EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
    return AF3EDMSampler(config)


@pytest.fixture
def mock_trajectory_context() -> StepParams:
    """Valid StepParams for trajectory-based sampling."""
    return StepParams(
        step_index=0,
        total_steps=10,
        t=torch.tensor([1.0]),
        dt=torch.tensor([-0.1]),
        noise_scale=torch.tensor([1.003]),
    )


# ============================================================================
# Mock scaler and reward fixtures
# ============================================================================


@pytest.fixture
def mock_step_scaler() -> MockStepScaler:
    """MockStepScaler with default step_size for guidance tests."""
    return MockStepScaler(step_size=0.1)


@pytest.fixture
def mock_gradient_reward() -> MockGradientRewardFunction:
    """MockGradientRewardFunction for guidance tests."""
    return MockGradientRewardFunction(gradient_scale=1.0)


# ============================================================================
# Mock processed structure fixture
# ============================================================================


@pytest.fixture
def mock_processed_structure(mock_wrapper: MockFlowModelWrapper) -> SampleworksProcessedStructure:
    """Mock SampleworksProcessedStructure for trajectory scaler tests."""
    num_atoms = mock_wrapper.num_atoms
    features = mock_wrapper.featurize({})
    atom_array = AtomArray(num_atoms)
    atom_array.coord = np.random.randn(num_atoms, 3).astype(np.float32)
    atom_array.set_annotation("element", np.array(["C"] * num_atoms))
    atom_array.set_annotation("b_factor", np.full(num_atoms, 20.0))
    atom_array.set_annotation("occupancy", np.ones(num_atoms))
    return SampleworksProcessedStructure(
        structure={},
        model_input=features,
        input_coords=torch.randn(1, num_atoms, 3),
        atom_array=atom_array,
        ensemble_size=1,
        reconciler=AtomReconciler.identity(num_atoms),
    )


@pytest.fixture
def perturbed_coords(
    converging_mock_wrapper: MockFlowModelWrapper, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Base structure and perturbed version for partial diffusion testing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (base_coords, perturbed_coords) with shape (1, num_atoms, 3)
    """
    torch.manual_seed(42)
    base = converging_mock_wrapper.target
    if base is None:
        raise ValueError("converging_mock_wrapper must provide target coordinates")
    perturbation = torch.randn_like(base) * 0.1
    return base, base + perturbation
