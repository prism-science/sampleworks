from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor


# Baked-in checkpoint paths (Docker image), ACTL shared-storage paths, and
# legacy fallbacks. Environment variables win when present.
_CHECKPOINT_ENV_VARS = {
    "boltz1": "BOLTZ1_CHECKPOINT",
    "boltz2": "BOLTZ2_CHECKPOINT",
    "rf3": "RF3_CHECKPOINT",
    "protenix": "PROTENIX_CHECKPOINT",
}

_CHECKPOINT_CANDIDATES = {
    "boltz1": [
        "/checkpoints/boltz1_conf.ckpt",
        "/mnt/diffuse-shared/raw/checkpoints/boltz1_conf.ckpt",
        "~/.boltz/boltz1_conf.ckpt",
    ],
    "boltz2": [
        "/checkpoints/boltz2_conf.ckpt",
        "/mnt/diffuse-shared/raw/checkpoints/boltz2_conf.ckpt",
        "~/.boltz/boltz2_conf.ckpt",
    ],
    "rf3": [
        "/checkpoints/rf3_foundry_01_24_latest.ckpt",
        "/mnt/diffuse-shared/raw/checkpoints/rf3_foundry_01_24_latest.ckpt",
        "~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt",
    ],
    "protenix": [
        "/checkpoints/protenix_base_default_v0.5.0.pt",
        "/mnt/diffuse-shared/raw/checkpoints/protenix_base_default_v0.5.0.pt",
        ".pixi/envs/protenix-dev/lib/python3.12/site-packages/release_data/checkpoint/protenix_base_default_v0.5.0.pt",
    ],
}


def _resolve_checkpoint(model_key: str) -> str:
    """Return the first checkpoint path that exists on disk for *model_key*.

    Model-specific environment variables from :data:`_CHECKPOINT_ENV_VARS` win
    when set. Otherwise, candidates from :data:`_CHECKPOINT_CANDIDATES` are
    tried in order, starting with baked-in ``/checkpoints/`` paths and then
    ACTL shared-storage and legacy development locations.
    """
    env_var = _CHECKPOINT_ENV_VARS.get(model_key)
    candidates = []
    if env_var and os.environ.get(env_var):
        candidates.append(os.environ[env_var])
    candidates.extend(_CHECKPOINT_CANDIDATES.get(model_key, []))
    for candidate in candidates:
        resolved = Path(candidate).expanduser()
        if resolved.exists():
            return str(resolved)
    # Nothing found – return the primary (baked-in) path so the error message
    # points the user to the expected location.
    resolved = candidates[0] if candidates else ""
    if not resolved:
        raise ValueError(
            f"Running guidance requires a model checkpoint for '{model_key}'. "
            f"Provide --model-checkpoint or bake checkpoints into /checkpoints/."
        )
    if not Path(resolved).exists():
        env_hint = _CHECKPOINT_ENV_VARS.get(model_key, "a checkpoint env var")
        raise ValueError(
            f"Model checkpoint for '{model_key}' was not found. Checked: {candidates}. "
            f"Provide --model-checkpoint or set {env_hint}."
        )

    return resolved


# ---------------------------------------------------------------------------
# Container-to-host path remapping (for job_metadata.json serialization)
# ---------------------------------------------------------------------------

_ENV_VAR_MAPPINGS = [
    # Order matters: most-specific container prefix first.
    ("SAMPLEWORKS_HOST_INPUT_DIR", "/data/inputs"),
    ("SAMPLEWORKS_HOST_RESULTS_DIR", "/data/results"),
    ("SAMPLEWORKS_HOST_DIR", "/data"),
]


def _remap_container_path(path_str: str) -> str:
    """Remap a container-internal path to its host equivalent.

    Uses environment variables to determine the mapping:

    * ``SAMPLEWORKS_HOST_INPUT_DIR``    replaces ``/data/inputs``
    * ``SAMPLEWORKS_HOST_RESULTS_DIR`` replaces ``/data/results``
    * ``SAMPLEWORKS_HOST_DIR``         replaces ``/data``  (fallback for single-mount setups)

    When none of these env vars are set, the path is returned unchanged.
    Paths that don't match any known container prefix (e.g. ``/checkpoints/``)
    are also returned unchanged.
    """
    env_pairs: list[tuple[str, str]] = []
    for env_var, container_prefix in _ENV_VAR_MAPPINGS:
        host_dir = os.environ.get(env_var)
        if host_dir is not None and host_dir.strip():
            host_dir = host_dir.strip()
            normalized_host = "/" if host_dir == "/" else host_dir.rstrip("/")
            env_pairs.append((container_prefix.rstrip("/"), normalized_host))

    for container_prefix, host_prefix in env_pairs:
        if path_str == container_prefix or path_str.startswith(container_prefix + "/"):
            result = host_prefix + path_str[len(container_prefix) :]
            return result[1:] if result.startswith("//") else result

    return path_str


def get_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve a model checkpoint path from an argparse namespace.

    Looks for a ``model_checkpoint`` attribute on *args*.
    Empty strings are treated as missing values.
    """
    value = getattr(args, "model_checkpoint", None)
    if value is not None and str(value).strip() != "":
        return str(value)

    return None


def validate_model_checkpoint(
    model: str | StructurePredictor,
    checkpoint: str | Path | None,
) -> str:
    """Validate and normalize the checkpoint path for ``model``.

    When *checkpoint* is ``None`` (no ``--model-checkpoint`` provided), the
    function auto-resolves by checking baked-in Docker paths first
    (``/checkpoints/``) and then legacy development paths.

    Returns
    -------
    str
        Absolute checkpoint path.

    Raises
    ------
    ValueError
        If checkpoint points to a directory.
    FileNotFoundError
        If checkpoint does not exist on disk.
    """
    # Auto-resolve when no explicit checkpoint was provided
    if checkpoint is None or str(checkpoint).strip() == "":
        model_key = str(model).lower().replace("structurepredictor.", "")
        resolved = _resolve_checkpoint(model_key)
        if not resolved:
            raise ValueError(
                f"Missing checkpoint for model '{model}'. "
                f"Provide --model-checkpoint or bake checkpoints into /checkpoints/."
            )
        checkpoint = resolved

    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for model '{model}' does not exist: {checkpoint_path}. "
            f"Provide a valid path via --model-checkpoint."
        )

    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint for model '{model}' must be a file, got: {checkpoint_path}")

    return str(checkpoint_path)


# Attributes set dynamically by add_*_args helpers that should be copied
# from a parsed argparse.Namespace onto a GuidanceConfig instance.
_DYNAMIC_ATTRS = [
    # pure guidance
    "step_size",
    "step_scaler_type",
    # fk steering
    "num_particles",
    "fk_resampling_interval",
    "fk_lambda",
    "num_gd_steps",
    "guidance_weight",
    "guidance_interval",
    # latent optimization (IT-opt) -- must be listed here or from_cli() drops the parsed
    # values and _run_guidance()'s getattr(args, ...) always sees the defaults (flags = no-ops).
    "which_latent",
    "learning_rate",
    "outer_steps",
    "anchor_weight",
    "max_grad_norm",
    "bond_length_weight",
    # model-specific
    "model_checkpoint",
    "method",
    "msa_path",
    "disable_chiral_features",
    "track_chiral_features",
    "protpardelle_config_path",
    # generic (overridable)
    "ensemble_size",
    "recycling_steps",
    "num_diffusion_steps",
]


@dataclass
class GuidanceConfig:
    """
    Class to hold guidance config arguments, compatible with argparse, but which
    also can do some basic validation.
    """

    # add basic arguments by default.
    protein: str
    structure: Path | str  # actually a path to a structure file
    density: Path | str
    model_name: str | StructurePredictor
    guidance_type: str | GuidanceType
    log_path: str
    output_dir: str = "output"
    partial_diffusion_step: int = 0
    loss_order: int = 2
    resolution: float | None = None
    device: str = ""
    gradient_normalization: bool = False
    em: bool = False
    guidance_start: int = -1
    augmentation: bool = False
    align_to_input: bool = False
    alignment_reverse_diffusion: bool | None = None
    recycling_steps: int | None = None
    num_diffusion_steps: int = 200

    # DO NOT remove the **kwargs, it is for compatibility with argparse.
    def add_argument(self, name: str, default: Any = None, **kwargs):
        """Add an argument to the guidance config, in a form compatible with argparse"""
        setattr(self, name.lstrip("-").replace("-", "_"), default)

    @classmethod
    def from_cli(
        cls,
        argv: list[str] | None = None,
        model_name: str | None = None,
        guidance_type: str | None = None,
    ) -> GuidanceConfig:
        """Parse CLI arguments and return a fully populated GuidanceConfig.

        When *model_name* and *guidance_type* are provided (e.g. from legacy
        scripts), they are used directly and ``--model`` / ``--guidance-type``
        are not required on the command line.  Otherwise they are parsed as
        required CLI arguments.
        """
        model_choices = [m.value for m in StructurePredictor]
        guidance_choices = [g.value for g in GuidanceType]
        model_preset = model_name is not None
        guidance_preset = guidance_type is not None

        if model_preset and model_name not in model_choices:
            raise ValueError(f"Unknown model type: {model_name}")
        if guidance_preset and guidance_type not in guidance_choices:
            raise ValueError(f"Unknown guidance type: {guidance_type}")

        # -- first pass: resolve model & guidance_type if not pre-set --------
        if not model_preset or not guidance_preset:
            pre = argparse.ArgumentParser(add_help=False)
            if not model_preset:
                pre.add_argument(
                    "--model",
                    dest="model_name",
                    type=str,
                    required=True,
                    choices=model_choices,
                    help="Structure prediction model",
                )
            if not guidance_preset:
                pre.add_argument(
                    "--guidance-type",
                    type=str,
                    required=True,
                    choices=guidance_choices,
                    help="Guidance method",
                )
            pre_args, _ = pre.parse_known_args(argv)
            model_name = model_name or pre_args.model_name
            guidance_type = guidance_type or pre_args.guidance_type

        if model_name is None or guidance_type is None:
            raise RuntimeError("CLI parsing did not resolve a model name and guidance type")

        # -- full parser -----------------------------------------------------
        parser = argparse.ArgumentParser(
            description=f"Run {guidance_type} guidance with {model_name}",
        )
        parser.add_argument(
            "--model",
            dest="model_name",
            type=str,
            default=model_name,
            choices=model_choices,
            help=argparse.SUPPRESS if model_preset else "Structure prediction model",
        )
        parser.add_argument(
            "--guidance-type",
            type=str,
            default=guidance_type,
            choices=guidance_choices,
            help=argparse.SUPPRESS if guidance_preset else "Guidance method",
        )
        parser.add_argument(
            "--protein",
            type=str,
            required=True,
            help="Protein identifier (must match naming used in grid search / evaluation)",
        )
        add_generic_args(parser)
        _MODEL_ARG_ADDERS[model_name](parser)
        _GUIDANCE_ARG_ADDERS[guidance_type](parser)

        args = parser.parse_args(argv)

        if model_preset and args.model_name != model_name:
            parser.error(
                f"This script is fixed to --model {model_name}."
                f" Use sampleworks-guidance for other models."
            )
        if guidance_preset and args.guidance_type != guidance_type:
            parser.error(
                f"This script is fixed to --guidance-type {guidance_type}."
                f" Use sampleworks-guidance for other guidance types."
            )

        config = cls(
            protein=args.protein,
            structure=args.structure,
            density=args.density,
            model_name=model_name,
            guidance_type=guidance_type,
            log_path=getattr(args, "log_path", None) or "",
            output_dir=args.output_dir,
            partial_diffusion_step=args.partial_diffusion_step,
            loss_order=args.loss_order,
            resolution=args.resolution,
            device=getattr(args, "device", "") or "",
            gradient_normalization=args.gradient_normalization,
            em=args.em,
            guidance_start=args.guidance_start,
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            alignment_reverse_diffusion=args.alignment_reverse_diffusion,
        )

        # __post_init__ already set defaults for model/guidance-specific
        # attrs; override with any explicit CLI values.
        for attr in _DYNAMIC_ATTRS:
            val = getattr(args, attr, None)
            if val is not None:
                setattr(config, attr, val)

        return config

    def __post_init__(self):
        """Set up guidance config for a given model and guidance type"""
        try:
            _GUIDANCE_ARG_ADDERS[self.guidance_type](self)
        except KeyError:
            raise ValueError(f"Unknown guidance type: {self.guidance_type}")

        try:
            _MODEL_ARG_ADDERS[self.model_name](self)
        except KeyError:
            raise ValueError(f"Unknown model type: {self.model_name}")

    def populate_config_for_guidance_type(self, job: JobConfig, args: argparse.Namespace):
        """Apply per-job grid-search values onto this guidance configuration."""
        checkpoint = get_checkpoint(args)
        if checkpoint is not None:
            self.model_checkpoint = checkpoint
        elif not getattr(self, "model_checkpoint", None):
            # Auto-resolve from baked-in /checkpoints/ or legacy fallback paths
            model_key = str(self.model_name).lower().replace("structurepredictor.", "")
            self.model_checkpoint = _resolve_checkpoint(model_key)

        if job.model_name == StructurePredictor.BOLTZ_2 and job.method:
            self.method = job.method

        if job.model_name == StructurePredictor.RF3:
            self.disable_chiral_features = getattr(args, "disable_chiral_features", False)
            self.track_chiral_features = getattr(args, "track_chiral_features", False)

        if job.scaler == GuidanceType.FK_STEERING:
            self.guidance_weight = job.gradient_weight
            self.num_gd_steps = job.gd_steps
            self.num_particles = args.num_particles
            self.fk_lambda = args.fk_lambda
            self.fk_resampling_interval = args.fk_resampling_interval
            self.ensemble_size = job.ensemble_size
        else:
            self.step_size = job.gradient_weight
            self.step_scaler_type = args.step_scaler_type
            self.ensemble_size = job.ensemble_size

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the guidance config, converting Path to strings.

        When host-path env vars are set, container-internal paths are remapped
        to their host equivalents so that
        ``job_metadata.json`` is reproducible outside the container.
        """
        output = self.__dict__.copy()
        output["density"] = _remap_container_path(str(self.density))
        output["structure"] = _remap_container_path(str(self.structure))
        output["output_dir"] = _remap_container_path(str(self.output_dir))
        output["log_path"] = _remap_container_path(str(self.log_path))
        return output

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state while migrating legacy pickles from ``model``."""
        migrated = state.copy()
        if "model" in migrated:
            migrated.setdefault("model_name", migrated.pop("model"))
        self.__dict__.update(migrated)


def add_generic_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments shared by all models and guidance methods."""
    parser.add_argument("--structure", type=str, required=True, help="Input structure")
    parser.add_argument("--density", type=str, required=True, help="Input density map")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--log-path", type=str, default=None, help="Log file path (default: output-dir/run.log)"
    )
    parser.add_argument(
        "--partial-diffusion-step",
        type=int,
        default=0,
        help="Diffusion step to start from",
    )
    parser.add_argument("--loss-order", type=int, default=2, choices=[1, 2], help="L1 or L2 loss")
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Map resolution in Angstroms (required for CCP4/MRC/MAP)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)")
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
    parser.add_argument("--em", action="store_true", help="Use EM scattering factors")
    parser.add_argument(
        "--guidance-start",
        type=int,
        default=-1,
        help="Step to start guidance (default: -1, starts immediately)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--align-to-input",
        action="store_true",
        help="Enable alignment to input",
    )
    parser.add_argument(
        "--alignment-reverse-diffusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Align the noisy state to the denoised prediction during reverse "
            "diffusion (described in Boltz-1 paper). Use "
            "--no-alignment-reverse-diffusion to disable. Default: enabled for "
            "Boltz, disabled for other models."
        ),
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=4,
        help="Ensemble size to generate (per particle for FK-steering)",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=None,
        help="Number of recycling steps for the model (default: model-specific)",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=200,
        help="Number of diffusion denoising steps (default: 200)",
    )


######################
# Guidance type specific arguments
######################
def add_pure_guidance_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to pure guidance sampling."""
    parser.add_argument("--step-size", type=float, default=0.1, help="Gradient step")
    parser.add_argument(
        "--step-scaler-type",
        type=str,
        default="noisespace",
        choices=["dataspace", "noisespace", "none"],
        help="Type of step scaler to use: dataspace (DataSpaceDPSScaler), noisespace "
        "(NoiseSpaceDPSScaler), or none (NoScalingScaler)",
    )


def add_fk_steering_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to Feynman-Kac steering."""
    parser.add_argument(
        "--num-particles",
        type=int,
        default=3,
        help="Number of particles for FK steering",
    )
    parser.add_argument(
        "--fk-resampling-interval",
        type=int,
        default=1,
        help="How often to apply resampling",
    )
    parser.add_argument(
        "--fk-lambda",
        type=float,
        default=1.0,
        help="Weighting factor for resampling",
    )
    parser.add_argument(
        "--num-gd-steps",
        type=int,
        default=1,
        help="Number of gradient descent steps on x0",
    )
    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=0.01,
        help="Weight for gradient descent guidance",
    )
    parser.add_argument(
        "--guidance-interval",
        type=int,
        default=1,
        help="How often to apply guidance",
    )


###########
# Model specific arguments
###########
def add_boltz2_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to Boltz2 guidance runs."""
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Boltz2 checkpoint (default: auto-resolved from /checkpoints/ or ~/.boltz/)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="X-RAY DIFFRACTION",
        help="Boltz2 sampling method",
    )


def add_protenix_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to Protenix guidance runs."""
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Protenix checkpoint (default: auto-resolved from /checkpoints/ or pixi env)",
    )


def add_boltz1_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to Boltz1 guidance runs."""
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Boltz1 checkpoint (default: auto-resolved from /checkpoints/ or ~/.boltz/)",
    )


def add_rf3_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to RF3 guidance runs."""
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to RF3 checkpoint (default: auto-resolved from /checkpoints/ or ~/.foundry/)",
    )
    parser.add_argument(
        "--msa-path",
        type=str,
        default=None,
        help="Path to MSA file (dict, JSON, or .a3m format)",
    )
    parser.add_argument(
        "--disable-chiral-features",
        action="store_true",
        help="Disable RF3 chiral gradient feature during guidance",
    )
    parser.add_argument(
        "--track-chiral-features",
        action="store_true",
        help="Log chiral gradient statistics at each denoising step",
    )


def add_protpardelle_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to Protpardelle guidance runs."""
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to Protpardelle checkpoint "
            "(default: auto-resolved from /checkpoints/ or pixi env)"
        ),
    )
    parser.add_argument(
        "--protpardelle-config-path",
        type=str,
        default=None,
        help="Path to the Protpardelle model config YAML (default: bundled cc89 config)",
    )


_MODEL_ARG_ADDERS: dict[str, Any] = {
    "boltz1": add_boltz1_specific_args,
    "boltz2": add_boltz2_specific_args,
    "protenix": add_protenix_specific_args,
    "rf3": add_rf3_specific_args,
    "protpardelle": add_protpardelle_specific_args,
}


def add_latent_opt_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments specific to inference-time latent optimization (IT-opt)."""
    parser.add_argument(
        "--which-latent",
        type=str,
        default="pair",
        choices=["single", "pair", "both"],
        help="Which trunk latent(s) to optimize: single (s), pair (z), or both",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Adam learning rate")
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=2,
        help="Number of optimization rounds (fresh diffusion noise per round)",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="On-manifold L2-to-baseline anchor weight for the optimized latent(s)",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Per-latent gradient-clip threshold"
    )
    parser.add_argument(
        "--bond-length-weight",
        type=float,
        default=5e-5,
        help="Weight of the coordinate-space bond-geometry penalty (bond-length + steric-clash "
        "hinges) added to the IT-opt loss. Default 5e-5 = smallest weight that fixes mean clash "
        "while keeping density fit and diversity; use 1e-3 for median clash 0; 0 disables it",
    )


_GUIDANCE_ARG_ADDERS: dict[str, Any] = {
    "pure_guidance": add_pure_guidance_args,
    "fk_steering": add_fk_steering_args,
    "latent_opt": add_latent_opt_args,
}


@dataclass
class JobConfig:
    """Resolved inputs and grid-search settings for one guidance job."""

    protein: str
    structure_path: Path | str
    density_path: Path | str
    resolution: float
    model_name: str
    scaler: str
    ensemble_size: int
    gradient_weight: float
    gd_steps: int
    method: str | None
    output_dir: str
    log_path: str


@dataclass
class JobResult:
    """Serializable status record produced after a guidance job finishes."""

    protein: str
    model_name: str
    method: str | None
    scaler: str
    ensemble_size: int
    gradient_weight: float
    gd_steps: int
    status: str
    exit_code: int
    runtime_seconds: float
    started_at: str
    finished_at: str
    log_path: str
    output_dir: str

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the job result.

        Mirrors :py:meth:`GuidanceConfig.as_dict`: when host-path env vars are set,
        ``output_dir`` and ``log_path`` are remapped to their host equivalents so
        ``job_metadata.json`` is reproducible outside the container.

        Returns
        -------
        dict[str, Any]
            JobResult fields with ``output_dir`` and ``log_path`` remapped via
            :py:func:`_remap_container_path`.
        """
        output = self.__dict__.copy()
        output["output_dir"] = _remap_container_path(str(self.output_dir))
        output["log_path"] = _remap_container_path(str(self.log_path))
        return output

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state while migrating legacy pickles from ``model``."""
        migrated = state.copy()
        if "model" in migrated:
            migrated.setdefault("model_name", migrated.pop("model"))
        self.__dict__.update(migrated)
