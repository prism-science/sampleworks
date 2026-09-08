from __future__ import annotations

import json
import os
import pickle
import traceback
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import save_structure
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import (
    RealSpaceRewardFunction,
    setup_scattering_params,
)
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.latent_optimization import LatentOptimization  # IT-opt wiring (added)
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import (
    DataSpaceDPSScaler,
    NoiseSpaceDPSScaler,
    NoScalingScaler,
)
from sampleworks.eval.occupancy_utils import extract_protein_and_occupancy
from sampleworks.utils.atom_array_utils import parse_structure
from sampleworks.utils.cif_utils import add_category_to_cif, resolve_mixed_hetatm_atom_altlocs
from sampleworks.utils.guidance_constants import (
    GuidanceType,
    StructurePredictor,
)
from sampleworks.utils.guidance_script_arguments import (
    _resolve_checkpoint,
    GuidanceConfig,
    JobResult,
    validate_model_checkpoint,
)
from sampleworks.utils.msa import MSAManager


# The following imports aren't compatible with each other and are supported in separate
# hatch/pixi envs
try:
    from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper
except ImportError:
    Boltz1Wrapper = None  # ty:ignore[invalid-assignment]
    logger.warning("Failed to import Boltz, hopefully you're running a different model")
try:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
except ImportError:
    ProtenixWrapper = None  # ty:ignore[invalid-assignment]
    logger.warning("Failed to import Protenix, hopefully you're running a different model")
try:
    from sampleworks.models.rf3.wrapper import RF3Wrapper
except ImportError:
    RF3Wrapper = None  # ty:ignore[invalid-assignment]
    logger.warning("Failed to import RF3, hopefully you're running a different model")
try:
    from sampleworks.models.protpardelle.wrapper import ProtpardelleWrapper
except (ImportError, OSError):  # OSError can arise from a missing model_params directory
    ProtpardelleWrapper = None  # ty:ignore[invalid-assignment]
    logger.warning(
        "Failed to import Protpardelle, hopefully you're running a different model. "
        "If you intended to use Protpardelle, please additionally check that the "
        "model_params directory exists. You may need to set the environment variable "
        "PROTPARDELLE_MODEL_PARAMS."
    )

from sampleworks.utils.torch_utils import try_gpu


def save_trajectory(
    scaler_type: str,
    trajectory,
    atom_array,
    output_dir,
    subdir_name,
    save_every=10,
):
    """Dispatch trajectory serialization to the handler for the selected scaler."""
    # IT-opt wiring (changed): this condition was `== GuidanceType.PURE_GUIDANCE`; we widened it to
    # also accept LATENT_OPT. Latent optimization reuses the pure-guidance trajectory writer because
    # its final sampling pass emits a trajectory with the same [ensemble, atoms, 3] layout that
    # _save_trajectory already expects, so no separate writer is needed for it.
    if scaler_type in (GuidanceType.PURE_GUIDANCE, GuidanceType.LATENT_OPT):
        _save_trajectory(trajectory, atom_array, output_dir, subdir_name, save_every)
    elif scaler_type == GuidanceType.FK_STEERING:
        _save_fk_steering_trajectory(trajectory, atom_array, output_dir, subdir_name, save_every)
    else:  # we shouldn't ever get here, since we can't have run guidance w/o this!
        raise ValueError(f"Invalid scaler type: {scaler_type}")


def _write_coords_into_array(
    array_copy: AtomArrayStack,
    coords: np.ndarray,
) -> None:
    """**Mutates** ``array_copy.coord`` in-place with trajectory coordinates.

    Coordinates must span all atoms in the array. Wrappers are responsible
    for producing model atom arrays with valid coordinates for every atom.
    """
    n_atoms_array = array_copy.coord.shape[-2]
    n_atoms_coords = coords.shape[-2]

    if n_atoms_coords != n_atoms_array:
        raise ValueError(
            f"Trajectory coords ({n_atoms_coords} atoms) don't match "
            f"atom array ({n_atoms_array} atoms)"
        )
    array_copy.coord = coords


def _save_trajectory(trajectory, atom_array, output_dir, subdir_name, save_every):
    """Save a pure-guidance coordinate trajectory as sampled multi-model CIFs."""
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    if not isinstance(atom_array, AtomArray):
        raise TypeError(
            "Can only save a trajectory of type AtomArray or "
            f"AtomArrayStack, was given {type(atom_array)}"
        )

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[0]
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        _write_coords_into_array(array_copy, coords.detach().numpy())
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def _save_fk_steering_trajectory(trajectory, atom_array, output_dir, subdir_name, save_every):
    """Save the first-particle FK-steering trajectory as sampled multi-model CIFs."""
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    if not isinstance(atom_array, AtomArray):
        raise TypeError(
            "Can only save a trajectory of type AtomArray or "
            f"AtomArrayStack, was given {type(atom_array)}"
        )

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[1]  # first dim is the particle dim
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        # we save only the first ensemble out of n_particles, since saving
        # each particle at every step would clog trajectory saving
        _write_coords_into_array(array_copy, coords[0].detach().numpy())
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def save_losses(losses, output_dir):
    """Write per-step guidance losses to ``losses.txt`` in ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "losses.txt", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(losses):
            if loss is not None:
                f.write(f"{i},{loss}\n")
            else:
                f.write(f"{i},NA\n")


def get_model_and_device(
    device_str: str,
    model_checkpoint_path: str | None,
    model_type: str,
    config: GuidanceConfig | None = None,
    model: Any = None,
) -> tuple[torch.device, Any]:
    """Validate a checkpoint, choose a device, and construct the model wrapper.

    Arguments:
        device_str: The device to use, e.g. "cuda:0" or "cpu".
        model_checkpoint_path: The path to the model checkpoint.
        model_type: The type of model to use.
        config: The configuration object, usually GuidanceConfig, from which extra
           model-specific settings are read, e.g. a path to a YAML config file.
        model: The model to use, if provided, helps to prevent re-loading the actual weights.

    Model-specific settings are read from ``config`` rather than passed as
    separate arguments: ``method`` is used only by the Boltz2 wrapper, and
    ``protpardelle_config_path`` only by the Protpardelle wrapper.
    """

    validated_checkpoint_path = validate_model_checkpoint(model_type, model_checkpoint_path)

    device = torch.device(device_str) if device_str else try_gpu()
    logger.debug(f"Using device: {device}")
    if model_type == StructurePredictor.PROTENIX:
        if ProtenixWrapper is None:
            raise ImportError("Protenix dependencies not installed")
        logger.debug(f"Loading Protenix model from {validated_checkpoint_path}")
        model_wrapper = ProtenixWrapper(
            checkpoint_path=validated_checkpoint_path, device=device, model=model
        )
    elif model_type == StructurePredictor.BOLTZ_1:
        if Boltz1Wrapper is None:
            raise ImportError("Boltz dependencies not installed")
        logger.debug(f"Loading Boltz1 model from {validated_checkpoint_path}")
        model_wrapper = Boltz1Wrapper(
            checkpoint_path=validated_checkpoint_path,
            use_msa_manager=True,
            device=device,
            model=model,
        )
    elif model_type == StructurePredictor.BOLTZ_2:
        if Boltz2Wrapper is None:
            raise ImportError("Boltz dependencies not installed")
        method = getattr(config, "method", None)
        if method is None:
            # TODO: make a useful error msg that includes options for method
            raise ValueError("Method must be specified for Boltz2")
        logger.debug(f"Loading Boltz2 model from {validated_checkpoint_path}")
        model_wrapper = Boltz2Wrapper(
            checkpoint_path=validated_checkpoint_path,
            use_msa_manager=True,
            device=device,
            method=method.upper(),
            model=model,
        )
    elif model_type == StructurePredictor.RF3:
        if RF3Wrapper is None:
            raise ImportError("RF3 dependencies not installed")
        logger.debug(f"Loading RF3 model from {validated_checkpoint_path}")
        model_wrapper = RF3Wrapper(
            checkpoint_path=validated_checkpoint_path,
            msa_manager=MSAManager(),
            device=device,
            model=model,
        )
    elif model_type == StructurePredictor.PROTPARDELLE:
        if ProtpardelleWrapper is None:
            raise ImportError("Protpardelle dependencies not installed")
        logger.debug(f"Loading Protpardelle model from {validated_checkpoint_path}")
        protpardelle_config_path = getattr(config, "protpardelle_config_path", None)
        config_path = protpardelle_config_path or files("sampleworks.data").joinpath(
            "cc89_epoch415.yaml"
        )
        model_wrapper = ProtpardelleWrapper(
            config_path=str(Path(config_path).expanduser().resolve()),
            checkpoint_path=validated_checkpoint_path,
            device=device,
            model=model,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # (pyright doesn't think Boltz1Wrapper etc are "Any")
    return device, model_wrapper


# TODO: further atomize for easier testing.
def get_reward_function_and_structure(
    density: str | Path,
    device: torch.device,
    em,
    loss_order,
    resolution,
    structure_path: str | Path,
) -> tuple[RealSpaceRewardFunction, dict[str, Any]]:
    """Load structure and density inputs and build the real-space reward function."""
    logger.debug(f"Loading structure from {structure_path}")
    structure_path = Path(structure_path)
    safe_structure_path = resolve_mixed_hetatm_atom_altlocs(structure_path)
    try:
        structure = parse_structure(safe_structure_path)
    finally:
        # resolve_mixed_hetatm_atom_altlocs always returns a Path (pinned by
        # test_always_returns_path), so comparing Paths cannot mistake the
        # original for the temporary copy.
        if safe_structure_path != structure_path:
            try:
                safe_structure_path.unlink()
            except OSError as error:
                logger.warning(f"Failed to remove temporary CIF: {safe_structure_path}: {error}")

    logger.debug(f"Loading density map from {density}")
    xmap = XMap.fromfile(density, resolution=resolution)

    logger.debug("Setting up scattering parameters")

    atom_array = structure["asym_unit"]
    scattering_params = setup_scattering_params(em_mode=em, device=device)

    selection_mask = atom_array.occupancy > 0
    n_selected = selection_mask.sum()
    logger.info(f"Selected {n_selected} atoms with occupancy > 0")

    logger.info("Creating reward function")
    reward_function = RealSpaceRewardFunction(
        xmap,
        scattering_params,
        selection_mask,
        em=em,
        loss_order=loss_order,
        device=device,
    )
    return reward_function, structure


def save_everything(
    args: GuidanceConfig,
    losses: list[Any],
    refined_structure: dict,
    traj_denoised: list[Any],
    traj_next_step: list[Any],
    scaler_type: str,
    final_state: torch.Tensor | None = None,
    model_atom_array: AtomArray | None = None,
) -> None:
    """Save everything: refined structure/ensemble CIF, trajectories, and losses.

    When `final_state` is provided, its coordinates are written into the
    `refined_structure` atom array (respecting the occupancy/NaN validity mask) before
    saving.  Both the denoised and next-step trajectories are saved as
    multi-model CIF files (subsampled every 10 steps via ``save_trajectory``).

    Parameters
    ----------
    args : GuidanceConfig
        The arguments for the guidance run. This method directly uses args.output_dir,
        and creates that directory if it does not exist. The result of args.as_dict() is
        written to a JSON file in the same directory, and inserted into the output CIF file.
    losses : list[Any]
        Per-step loss values (may contain ``None`` entries for unguided steps).
    refined_structure : dict
        Atomworks structure dict whose ``"asym_unit"`` is used as the template
        for saving.
    traj_denoised : list[Any]
        Denoised-prediction trajectory tensors, one per diffusion step.
    traj_next_step : list[Any]
        Next-step (noisy) trajectory tensors, one per diffusion step.
    scaler_type : str
        Scaler/guidance identifier, forwarded to ``save_trajectory`` for
        scaler-specific handling. # TODO: handle more gracefully
    final_state : torch.Tensor | None
        Final coordinates with shape ``(ensemble, atoms, 3)``.  If ``None``,
        the `refined_structure`'s existing coordinates are saved as-is.
    model_atom_array : AtomArray | None
        Optional model-space atom template. When provided (mismatch runs),
        this template is used for final structure and trajectory saving.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    base_atom_array = ensure_atom_array_stack(refined_structure["asym_unit"])[0]

    # Use model's internal atom accounting template for mismatch runs when available
    # Wrappers must guarantee model atom arrays have valid coords and occupancy.
    atom_array_for_saving: AtomArray = (
        model_atom_array if model_atom_array is not None else base_atom_array
    )

    if final_state is not None:
        ensemble_size = final_state.shape[0]

        ensemble_array = stack([atom_array_for_saving.copy() for _ in range(ensemble_size)])
        _write_coords_into_array(ensemble_array, final_state.detach().cpu().numpy())
        atom_array = ensemble_array
    else:
        atom_array = base_atom_array

    metadata = args.as_dict()

    final_structure = CIFFile()
    set_structure(final_structure, atom_array)
    add_category_to_cif(final_structure, metadata, category_name="sampleworks")
    final_structure.write(str(output_dir / "refined.cif"))

    # job_metadata.json (config + JobResult) is written by run_guidance after this returns;
    # don't duplicate it here.

    # Two calls to save_trajectory, very similar, but saving different trajectories!
    save_trajectory(
        scaler_type,
        traj_denoised,  # <--- the difference is here!
        atom_array_for_saving,
        output_dir,
        "denoised",
        save_every=10,
    )
    save_trajectory(
        scaler_type,
        traj_next_step,  # <--- and here!
        atom_array_for_saving,
        output_dir,
        "next_step",
        save_every=10,
    )
    save_losses(losses, output_dir)

    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        logger.info(f"\nFinal loss: {valid_losses[-1]:.6f}")
        logger.info(f"Initial loss: {valid_losses[0]:.6f}")
        logger.info(f"Loss reduction: {valid_losses[0] - valid_losses[-1]:.6f}")

    logger.info(f"\nResults saved to {output_dir}/")


#####################
# Methods for running model guidance in separate processes, avoiding reloading of the model.
#####################
# These args are passed from run_grid_search.py via GuidanceConfig.
def run_guidance(args: GuidanceConfig, guidance_type: str, model_wrapper, device) -> JobResult:
    """Wrapper around ``_run_guidance`` to redirect logs and generate a JobResult.

    Parameters
    ----------
    args : GuidanceConfig
        Configuration for the guidance run.
    guidance_type : str
        Type of guidance/scaler to apply.
    model_wrapper
        Loaded model wrapper instance.
    device
        Torch device to run on.

    Returns
    -------
    JobResult
        Result of the guidance run including status and timing.
    """

    log_path = args.log_path or os.path.join(args.output_dir, "run.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # just in case log_path does not go to args.output_dir, make sure the latter exists
    os.makedirs(args.output_dir, exist_ok=True)

    # separate logs for each guidance run
    handle = logger.add(
        log_path, level="INFO", filter=lambda rec: rec["extra"].get("special", False) is True
    )
    started_at = datetime.now()
    try:
        with logger.contextualize(special=True):
            _run_guidance(args, guidance_type, model_wrapper, device)
        logger.info("Guidance run successfully!")
        job_result = get_job_result(args, device, started_at, datetime.now(), 0, "success")
    except Exception as e:
        logger.error(f"Error running guidance: {e} consult logs ({log_path}) for real errors.")
        logger.error(traceback.format_exc())
        job_result = get_job_result(args, device, started_at, datetime.now(), 1, "failed")
    finally:
        logger.remove(handle)

    _write_job_metadata(args.output_dir, args, job_result)
    return job_result


def _three_state_resolver(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


# "guidance_type" is also called "scaler" in many places
def _run_guidance(args: GuidanceConfig, guidance_type: str, model_wrapper, device):
    """Run one configured guidance trajectory and save its outputs."""
    reward_function, structure = get_reward_function_and_structure(
        args.density,  # str/path to a map file.
        device,  # this needs to come from the global context, not the args object.
        args.em,
        args.loss_order,
        args.resolution,
        args.structure,  # path/string to a structure file.
    )

    # Determine model type from wrapper class name
    wrapper_class_name = model_wrapper.__class__.__name__
    is_boltz = "Boltz" in wrapper_class_name

    # Annotate structure with model-specific configuration (including recycling_steps)
    # See https://github.com/prism-science/sampleworks/issues/192 for a plan to organize this
    # better.
    recycling_steps = getattr(args, "recycling_steps", None)
    if recycling_steps is not None and recycling_steps <= 0:
        raise ValueError("recycling_steps must be > 0")
    if args.num_diffusion_steps is not None and args.num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be > 0")

    edm_sampler_kwargs = {}  # i.e. use defaults.
    if "Protenix" in wrapper_class_name:
        from sampleworks.models.protenix.wrapper import annotate_structure_for_protenix

        structure = annotate_structure_for_protenix(
            structure,
            recycling_steps=recycling_steps,
            # Disable diffusion shared-vars cache for LATENT_OPT so gradients can
            # flow to z_trunk; cached tensors can otherwise become stale.
            # Keep cache enabled for other guidance types.
            enable_diffusion_shared_vars_cache=(guidance_type != GuidanceType.LATENT_OPT),
        )
    elif "RF3" in wrapper_class_name:
        from sampleworks.models.rf3.wrapper import annotate_structure_for_rf3

        structure = annotate_structure_for_rf3(
            structure,
            recycling_steps=recycling_steps,
            msa_path=args.msa_path,
            disable_chiral_features=args.disable_chiral_features,
            track_chiral_features=args.track_chiral_features,
        )
    elif "Boltz" in wrapper_class_name:
        from sampleworks.models.boltz.wrapper import process_structure_for_boltz

        # Boltz preprocessing writes manifest/NPZ/MSA files as a side effect.
        # Keep those under the per-job output directory so concurrent grid jobs
        # for the same protein do not race on a shared metadata-derived path.
        structure = process_structure_for_boltz(
            structure,
            out_dir=args.output_dir,
            recycling_steps=recycling_steps,
        )
    elif "Protpardelle" in wrapper_class_name:
        from sampleworks.models.protpardelle.wrapper import annotate_structure_for_protpardelle

        structure = annotate_structure_for_protpardelle(structure)
        edm_sampler_kwargs = {
            "s_max": 80,
            "s_min": 0.001,
            "gamma_0": 0.08,
            "gamma_min": 0.00,
            "sigma_data": 10.3,
            "step_scale": 1.0,
        }
    else:
        raise ValueError(f"Unknown model wrapper class: {wrapper_class_name}")

    use_alignment_for_reverse_diffusion = _three_state_resolver(
        args.alignment_reverse_diffusion, is_boltz
    )

    # Create sampler with model-appropriate settings
    sampler_config = EDMSamplerConfig(
        device=str(device),
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        alignment_reverse_diffusion=use_alignment_for_reverse_diffusion,
        **edm_sampler_kwargs,
    )
    sampler = AF3EDMSampler(
        config=sampler_config,
    )

    # Create step scaler for gradient-based guidance.
    # TODO: unify this arg, no need for both of them now
    step_size = getattr(args, "step_size", None)
    if step_size is None:
        step_size = getattr(args, "guidance_weight", 0.01)

    step_scaler_type = getattr(args, "step_scaler_type", "noisespace")
    if step_scaler_type == "dataspace":
        step_scaler = DataSpaceDPSScaler(
            step_size=step_size,
            gradient_normalization=args.gradient_normalization,
        )
    elif step_scaler_type == "noisespace":
        step_scaler = NoiseSpaceDPSScaler(
            step_size=step_size,
            gradient_normalization=args.gradient_normalization,
        )
    elif step_scaler_type == "none":
        step_scaler = NoScalingScaler()
    else:
        raise ValueError(f"Invalid step_scaler_type: {step_scaler_type}")

    num_steps = args.num_diffusion_steps

    if guidance_type == GuidanceType.PURE_GUIDANCE:
        logger.info("Initializing pure guidance")

        # TODO: these should be fractions in the args directly
        guidance_t_start = args.guidance_start / num_steps if args.guidance_start > 0 else 0.0
        t_start = args.partial_diffusion_step / num_steps if args.partial_diffusion_step else 0.0

        guidance = PureGuidance(
            ensemble_size=args.ensemble_size,
            num_steps=num_steps,
            t_start=t_start,
            guidance_t_start=guidance_t_start,
        )

        logger.info(f"Running pure guidance using model with hash {model_wrapper.model.__hash__()}")
        result = guidance.sample(
            structure=structure,
            model=model_wrapper,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward_function,
        )

        refined_structure = result.structure
        losses = result.losses if result.losses else []
        traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
        traj_next_step = list(result.trajectory) if result.trajectory else []

    elif guidance_type == GuidanceType.FK_STEERING:
        logger.info("Initializing Feynman-Kac steering")

        # TODO: same as above
        gs = args.guidance_start
        guidance_start_fraction = gs / num_steps if gs > 0 else 0.0
        pd = args.partial_diffusion_step
        t_start = pd / num_steps if pd else 0.0

        guidance = FKSteering(
            ensemble_size=args.ensemble_size,
            num_steps=num_steps,
            resampling_interval=args.fk_resampling_interval,
            fk_lambda=args.fk_lambda,
            guidance_t_start=guidance_start_fraction,
            t_start=t_start,
        )

        logger.info("Running FK steering")
        result = guidance.sample(
            structure=structure,
            model=model_wrapper,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward_function,
            num_particles=args.num_particles,
        )

        refined_structure = result.structure
        losses = result.losses if result.losses else []
        traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
        traj_next_step = list(result.trajectory) if result.trajectory else []

    # ----- IT-opt wiring (added): the inference-time latent optimization branch (LATENT_OPT) -----
    # This branch belongs to _run_guidance() and is the entry point for latent optimization. Whereas
    # pure guidance and FK steering steer the atomic coordinates, latent optimization instead
    # optimizes the frozen model's cached trunk latents (the single representation s and/or the pair
    # representation z) against the reward, then samples with those latents held fixed. We read the
    # knobs off the config as the other branches do and hand the optimize-then-sample loop to
    # LatentOptimization.
    elif guidance_type == GuidanceType.LATENT_OPT:
        logger.info("Initializing inference-time latent optimization (IT-opt)")

        # We import the representation-name maps inside this branch so the model-adapter dependency
        # stays local to the only code that needs it. The attribute that stores each representation
        # differs per model (for example "s_trunk"/"z_trunk" on Protenix and RF3 versus "s"/"z" on
        # Boltz), so we look the names up by model instead of hard-coding them.
        from sampleworks.models.latent_adapter import (
            DEFAULT_PAIR_REP_ATTR,
            DEFAULT_SINGLE_REP_ATTR,
        )

        # LatentOptimization expects guidance start as a fraction of the schedule, but the config
        # carries it as an integer step count, so we convert it here and default to optimizing from
        # the first step.
        guidance_t_start = args.guidance_start / num_steps if args.guidance_start > 0 else 0.0
        which_latent = args.which_latent  # This is "single", "pair", or "both".
        anchor_weight = args.anchor_weight
        bond_length_weight = args.bond_length_weight
        # GuidanceConfig exposes the model as `model_name` (not `args.model`); normalize to the
        # lowercase key the DEFAULT_*_REP_ATTR maps use (as checkpoint resolution does below).
        model_key = str(args.model_name).lower().replace("structurepredictor.", "")
        try:
            single_attr = DEFAULT_SINGLE_REP_ATTR[model_key]
            pair_attr = DEFAULT_PAIR_REP_ATTR[model_key]
        except KeyError as e:
            raise ValueError(
                "Latent optimization has no latent-attribute names registered for model "
                f"{model_key!r}."
            ) from e

        guidance = LatentOptimization(
            ensemble_size=args.ensemble_size,
            num_steps=num_steps,
            guidance_t_start=guidance_t_start,
            outer_steps=args.outer_steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            optimize_single=which_latent in ("single", "both"),
            optimize_pair=which_latent in ("pair", "both"),
            single_attr=single_attr,
            pair_attr=pair_attr,
            anchor_weight_single=anchor_weight if which_latent in ("single", "both") else 0.0,
            anchor_weight_pair=anchor_weight if which_latent in ("pair", "both") else 0.0,
            bond_length_weight=bond_length_weight,
        )

        logger.info(f"Running latent optimization ({which_latent}) on model {model_key}")
        # We still pass step_scaler so this call matches the signature the other guidance scalers
        # use, but LatentOptimization ignores it -- v1 steers only through the latents.
        result = guidance.sample(
            structure=structure,
            model=model_wrapper,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward_function,
        )

        refined_structure = result.structure
        losses = result.losses if result.losses else []
        traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
        traj_next_step = list(result.trajectory) if result.trajectory else []
    else:
        logger.error(f"Unknown guidance type: {guidance_type}")
        raise TypeError("Unknown guidance type!")

    model_atom_array = result.metadata.get("model_atom_array") if result.metadata else None

    save_everything(
        args,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        guidance_type,
        final_state=torch.as_tensor(result.final_state),
        model_atom_array=model_atom_array,
    )

    if hasattr(model_wrapper, "_chiral_grad_stats") and model_wrapper._chiral_grad_stats:
        stats_path = Path(args.output_dir) / "chiral_grad_stats.json"
        with open(stats_path, "w") as f:
            json.dump(model_wrapper._chiral_grad_stats, f, indent=2)
        logger.info(f"Saved chiral gradient stats to {stats_path}")


def _write_job_metadata(
    output_dir: str | Path,
    args: GuidanceConfig,
    job_result: JobResult,
) -> None:
    """Write ``job_metadata.json`` merging GuidanceConfig and JobResult fields.

    Both :py:meth:`GuidanceConfig.as_dict` and :py:meth:`JobResult.as_dict` apply
    container-to-host path remapping, so the merged file is consistent regardless of
    where the run executed.

    Parameters
    ----------
    output_dir : str | Path
        Directory in which to write ``job_metadata.json``. Created if missing.
    args : GuidanceConfig
        Configuration used for the guidance run. Provides the base metadata payload.
    job_result : JobResult
        Completed job result. Its fields (notably ``started_at``, ``finished_at``,
        ``runtime_seconds``, ``status``, ``exit_code``) are merged on top of the
        ``GuidanceConfig`` payload.
    """
    metadata = args.as_dict()
    metadata.update(job_result.as_dict())
    _, altloc_occupancies = extract_protein_and_occupancy(str(args.protein))
    if altloc_occupancies:
        metadata["altloc_occupancies"] = altloc_occupancies
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "job_metadata.json", "w") as fp:
        json.dump(metadata, fp)


def epoch_seconds(time_to_convert: datetime) -> float:
    """Convert a :class:`datetime.datetime` to seconds since the Unix epoch."""
    return (time_to_convert - datetime(1970, 1, 1)).total_seconds()


def get_job_result(
    args: GuidanceConfig,
    device: torch.device,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int,
    status: str,
) -> JobResult:
    """Build the serializable result record for a completed guidance job."""
    start_time = epoch_seconds(started_at)
    end_time = epoch_seconds(ended_at)
    result = JobResult(
        protein=args.protein,
        model_name=args.model_name,
        method=getattr(args, "method", None),
        scaler=args.guidance_type,
        ensemble_size=args.ensemble_size,
        gradient_weight=getattr(args, "guidance_weight", -1.0),
        gd_steps=getattr(args, "num_gd_steps", -1),
        status=status,
        exit_code=exit_code,
        runtime_seconds=round(end_time - start_time, 2),
        started_at=started_at.isoformat(),
        finished_at=ended_at.isoformat(),
        log_path=args.log_path or os.path.join(args.output_dir, "run.log"),
        output_dir=args.output_dir,
    )
    return result


def run_guidance_job_queue(job_queue_path: str) -> list[JobResult]:
    """Load a pickled job queue, reuse one model wrapper, and run all jobs."""
    with open(job_queue_path, "rb") as fp:
        job_queue: list[GuidanceConfig] = pickle.load(fp)

    template_job = job_queue[0]
    if template_job.model_checkpoint is None or template_job.model_checkpoint == "":
        # Auto-resolve from baked-in /checkpoints/ or legacy fallback paths
        model_key = str(template_job.model_name).lower().replace("structurepredictor.", "")
        resolved = _resolve_checkpoint(model_key)  # will raise if not found
        template_job.model_checkpoint = resolved
        # Propagate to all jobs in the queue
        for job in job_queue:
            job.model_checkpoint = resolved

    logger.info(f"Running {len(job_queue)} jobs, using {template_job} as a setup template")
    device, model_wrapper = get_model_and_device(
        str(template_job.device),
        template_job.model_checkpoint,
        template_job.model_name,
        config=template_job,
    )
    job_results = []
    for i, job in enumerate(job_queue):
        logger.info(f"Running job {i + 1}/{len(job_queue)}: {job}")

        job_result = run_guidance(job, job.guidance_type, model_wrapper, device)

        job_results.append(job_result)
        torch.cuda.empty_cache()  # just in case

    if hasattr(model_wrapper, "msa_manager") and model_wrapper.msa_manager is not None:
        # reports the number of API calls and cache hits
        model_wrapper.msa_manager.report_on_usage()
    else:
        logger.warning(
            "No MSA manager found, cannot report on MSA usage. "
            "(why aren't you using an MSAManager?)"
        )

    return job_results
