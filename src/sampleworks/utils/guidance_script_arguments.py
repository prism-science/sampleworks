from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sampleworks.core.rewards.config import REWARD_OPTIONS_KEY, RewardConfig
from sampleworks.core.rewards.options import option_type
from sampleworks.core.rewards.registry import get_reward_spec, reward_type_names
from sampleworks.utils.guidance_constants import GuidanceType, Rewards, StructurePredictor


# Reward used when a run does not say which one it wants. Keeps every command
# line written before rewards were selectable working unchanged.
DEFAULT_REWARD_TYPE = Rewards.REAL_SPACE_DENSITY

# Namespace prefix for generated reward-option flags, so they cannot collide with
# a model or guidance argument of the same name.
_REWARD_OPTION_PREFIX = "reward_option_"

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
    density: Path | str | None
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
    # Rewards this run scores against, as the {reward: {weight, reward_options}}
    # mapping of issue #358. Kept as plain primitives, not a RewardConfig: these
    # configs are pickled into job queues and read back by workers that may be
    # running a different sampleworks build. Left empty, __post_init__ fills it
    # in from the density fields below, which is how grid search and older
    # pickles keep working.
    reward_config: dict[str, Any] = field(default_factory=dict)

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

        # -- first pass: resolve model, guidance_type, and the reward selection --
        # The reward decides which option flags exist, so it has to be known before
        # the real parser is built, exactly like the model and the guidance type.
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
        add_reward_selection_args(pre)
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
        add_reward_selection_args(parser)
        # A configuration file names its own rewards, so the per-option flags of a
        # single reward would be ambiguous next to it: registering none of them
        # keeps --help honest about what this invocation accepts.
        if pre_args.reward_config is None:
            add_reward_args(parser, pre_args.reward_type)
        _MODEL_ARG_ADDERS[model_name](parser)
        _GUIDANCE_ARG_ADDERS[guidance_type](parser)

        args = parser.parse_args(argv)

        # The file names its own rewards; taking a --reward-type alongside it would
        # mean silently ignoring one of the two. --reward-type always has a default,
        # so "was it passed" has to be answered from the command line itself.
        given = argv if argv is not None else sys.argv[1:]
        if args.reward_config is not None and any(
            token.split("=", 1)[0] == "--reward-type" for token in given
        ):
            parser.error(
                "--reward-type and --reward-config are alternatives: the configuration "
                "file already names the rewards it configures."
            )
        reward_config = cls._reward_config_from_args(parser, args)

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
            density=None,  # mirrored from the reward configuration in __post_init__
            model_name=model_name,
            guidance_type=guidance_type,
            log_path=getattr(args, "log_path", None) or "",
            output_dir=args.output_dir,
            partial_diffusion_step=args.partial_diffusion_step,
            device=getattr(args, "device", "") or "",
            gradient_normalization=args.gradient_normalization,
            guidance_start=args.guidance_start,
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            alignment_reverse_diffusion=args.alignment_reverse_diffusion,
            reward_config=reward_config.to_mapping(),
        )

        # __post_init__ already set defaults for model/guidance-specific
        # attrs; override with any explicit CLI values.
        for attr in _DYNAMIC_ATTRS:
            val = getattr(args, attr, None)
            if val is not None:
                setattr(config, attr, val)

        return config

    @staticmethod
    def _reward_config_from_args(
        parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> RewardConfig:
        """Resolve the run's rewards from either CLI surface.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to report user errors through, so they read as usage errors.
        args : argparse.Namespace
            Parsed arguments.

        Returns
        -------
        RewardConfig
            The rewards this run scores against.
        """
        if args.reward_config is None:
            config = RewardConfig.single(
                args.reward_type, **reward_options_from_args(args)
            ).with_effective_options()
        else:
            try:
                config = RewardConfig.from_file(args.reward_config).with_effective_options()
            except (OSError, ValueError) as error:
                parser.error(f"--reward-config {args.reward_config}: {error}")

        # Report a missing input as a usage error now, rather than after a model
        # has been loaded. The reward builders check this too.
        missing = config.missing_required_options()
        if missing:
            parser.error(
                "; ".join(
                    f"the {reward} reward requires "
                    + ", ".join("--" + option.replace("_", "-") for option in options)
                    for reward, options in missing.items()
                )
            )

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

        self._reconcile_reward_config()

    def _reconcile_reward_config(self):
        """Keep ``reward_config`` and the flat density fields agreeing with each other.

        The density reward's options predate the reward configuration and are still
        the shape grid search builds configs in, ``job_metadata.json`` records, and
        the evaluation scripts read back. So the two representations are kept in
        sync in one place, in whichever direction has the information: a config
        built without ``reward_config`` (grid search, an older pickle) derives it
        from the flat fields, and one built with it mirrors the density options
        back out.
        """
        if not self.reward_config:
            options = {
                "density": None if self.density is None else str(self.density),
                "resolution": self.resolution,
                "loss_order": self.loss_order,
                "em": self.em,
            }
            self.reward_config = RewardConfig.single(
                DEFAULT_REWARD_TYPE,
                **{name: value for name, value in options.items() if value is not None},
            ).to_mapping()
            return

        density_options = self.reward_config.get(DEFAULT_REWARD_TYPE.value, {}).get(
            REWARD_OPTIONS_KEY, {}
        )
        self.density = density_options.get("density")
        self.resolution = density_options.get("resolution")
        self.loss_order = density_options.get("loss_order", 2)
        self.em = density_options.get("em", False)

    def resolved_reward_config(self) -> RewardConfig:
        """Return this run's rewards as a validated :class:`RewardConfig`.

        Returns
        -------
        RewardConfig
            Parsed from the stored mapping.
        """
        return RewardConfig.from_mapping(self.reward_config)

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

        ``reward_config`` is emitted as a JSON string rather than a nested
        mapping: this dictionary is also written into the output CIF as the
        ``sampleworks`` category, where a nested value would be read as a column
        of rows and produce a broken category.
        """
        output = self.__dict__.copy()
        output["density"] = (
            None if self.density is None else _remap_container_path(str(self.density))
        )
        output["structure"] = _remap_container_path(str(self.structure))
        output["output_dir"] = _remap_container_path(str(self.output_dir))
        output["log_path"] = _remap_container_path(str(self.log_path))
        output["reward_type"] = ",".join(self.reward_config)
        output["reward_config"] = json.dumps(
            self.resolved_reward_config().remapped_paths(_remap_container_path)
        )
        return output

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state, migrating pickles written before ``model_name`` and rewards.

        Job queues are pickled by whichever build submitted them and unpickled by
        the worker, so a config from an older build has to keep working: it names
        the model ``model``, and has no ``reward_config`` -- only the flat density
        fields the reward configuration is reconciled with.
        """
        migrated = state.copy()
        if "model" in migrated:
            migrated.setdefault("model_name", migrated.pop("model"))
        migrated.setdefault("reward_config", {})
        self.__dict__.update(migrated)
        self._reconcile_reward_config()


def add_reward_selection_args(parser: argparse.ArgumentParser):
    """Add the two ways of choosing rewards: one by name, or several from a file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add ``--reward-type`` and ``--reward-config`` to.
    """
    parser.add_argument(
        "--reward-type",
        type=str,
        default=DEFAULT_REWARD_TYPE.value,
        choices=reward_type_names(),
        help=f"Reward to guide with (default: {DEFAULT_REWARD_TYPE.value}). Its options are "
        "listed below. Use --reward-config to combine several rewards.",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="Reward configuration file (.yaml/.json/.toml) mapping each reward to its "
        "weight and reward_options. Takes the place of --reward-type and the "
        "per-reward flags, and is the only way to combine rewards.",
    )


def add_reward_args(parser: argparse.ArgumentParser, reward: Rewards | str):
    """Add the CLI flags for one reward's options, derived from its option schema.

    Only the selected reward's flags are registered, so a flag belonging to a
    different reward is rejected by argparse rather than silently ignored. Every
    flag defaults to None here: the option schema owns the real defaults, and
    "not passed" has to stay distinguishable from "passed the default" so a
    configuration file can be layered underneath.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the reward's flags to.
    reward : Rewards | str
        The selected reward type.
    """
    spec = get_reward_spec(reward)
    group = parser.add_argument_group(f"{spec.name.value} reward options", spec.description)

    for option in dataclasses.fields(spec.options_cls):
        value_type = option_type(spec.options_cls, option.name)
        metadata = option.metadata
        help_text = metadata["help"]
        if option.default is not None:
            help_text = f"{help_text} (default: {option.default})"

        kwargs: dict[str, Any] = {
            "dest": f"{_REWARD_OPTION_PREFIX}{option.name}",
            "default": None,
            "help": help_text,
        }
        if value_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            if not metadata["choices"]:
                # Otherwise the prefixed dest becomes the metavar and --help reads
                # "--mtzfile REWARD_OPTION_MTZFILE". Options with choices are left
                # alone so argparse can show the choices in their place.
                kwargs["metavar"] = option.name.upper()
            if metadata["json_arg"]:
                kwargs["type"] = json.loads
            elif typing.get_origin(value_type) is list:
                kwargs["type"] = typing.get_args(value_type)[0]
                kwargs["nargs"] = "+"
            else:
                kwargs["type"] = value_type

        if metadata["choices"] and value_type is not bool:
            kwargs["choices"] = list(metadata["choices"])

        group.add_argument("--" + option.name.replace("_", "-"), **kwargs)


def reward_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the reward options that were actually passed on the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict[str, Any]
        Option name to value, omitting everything left unset.
    """
    return {
        name.removeprefix(_REWARD_OPTION_PREFIX): value
        for name, value in vars(args).items()
        if name.startswith(_REWARD_OPTION_PREFIX) and value is not None
    }


def add_generic_args(parser: argparse.ArgumentParser | GuidanceConfig):
    """Add CLI arguments shared by all models and guidance methods."""
    parser.add_argument("--structure", type=str, required=True, help="Input structure")
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
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)")
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
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

_GUIDANCE_ARG_ADDERS: dict[str, Any] = {
    "pure_guidance": add_pure_guidance_args,
    "fk_steering": add_fk_steering_args,
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
