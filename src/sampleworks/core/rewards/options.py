"""Per-reward option schemas.

Each reward type declares its configurable options once, here, as a frozen
dataclass. Everything else is derived from that declaration: the CLI flags
(:func:`sampleworks.utils.guidance_script_arguments.add_reward_args`), the
``reward_options`` schema of a reward configuration file
(:mod:`sampleworks.core.rewards.config`), container path remapping in
``GuidanceConfig.as_dict``, and the "unknown option" error messages.

Option names are the configuration-file keys, and the CLI flag for an option is
always its name with underscores as dashes -- ``loss_order`` is ``--loss-order``
and nothing else. Keeping that mapping mechanical is what lets one declaration
serve every consumer.

This module deliberately imports nothing heavy. The registry has to be
importable in every pixi environment to render ``--help`` or validate a config,
while the reward implementations themselves pull in environment-specific stacks
(``SFC_Torch``, the vendored qFit code); those are only imported when a reward is
actually built.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any


def opt(
    default: Any,
    *,
    help: str,
    choices: tuple[Any, ...] | None = None,
    path: bool = False,
    json_arg: bool = False,
) -> Any:
    """Declare one reward option.

    Parameters
    ----------
    default
        Value used when neither the CLI nor a configuration file sets the option.
        The dataclass owns every default; the argparse layer defaults to ``None``
        so that "not passed" stays distinguishable from "passed the default".
    help
        Help text for the generated CLI flag.
    choices
        Allowed values, enforced by argparse and by config-file validation.
    path
        The value is a filesystem path, so it is remapped between container and
        host paths when the run configuration is serialized.
    json_arg
        The value is a JSON document on the command line (for dict-valued escape
        hatches such as ``sfcalculator_kwargs``).

    Returns
    -------
    Any
        A :func:`dataclasses.field` carrying the metadata above.
    """
    return field(
        default=default,
        metadata={"help": help, "choices": choices, "path": path, "json_arg": json_arg},
    )


@dataclass(frozen=True)
class RealSpaceDensityOptions:
    """Options for :class:`~sampleworks.core.rewards.real_space_density.RealSpaceRewardFunction`."""

    density: str | None = opt(None, help="Input density map (CCP4/MRC/MAP or MTZ)", path=True)
    resolution: float | None = opt(None, help="Map resolution in Angstroms")
    loss_order: int = opt(2, help="L1 or L2 loss", choices=(1, 2))
    em: bool = opt(False, help="Use electron (cryo-EM) scattering factors")


@dataclass(frozen=True)
class StructureFactorOptions:
    """Options for the structure-factor reward.

    See :class:`~sampleworks.core.rewards.structure_factor.StructureFactorRewardFunction`.
    """

    mtzfile: str | None = opt(None, help="MTZ holding the target amplitudes", path=True)
    expcolumns: list[str] | None = opt(
        None,
        help="MTZ amplitude and sigma column names, e.g. --expcolumns FP SIGFP "
        "(default: auto-detect, which requires exactly one of each)",
    )
    resolution: float | None = opt(
        None, help="High-resolution limit (dmin) in Angstroms (default: the MTZ's own)"
    )
    scattering_factor_mode: str = opt(
        "xray", help="SFcalculator scattering mode", choices=("xray", "cryoem")
    )
    bulk_solvent: str = opt(
        "off",
        help="Bulk-solvent treatment: off (score |Fprotein|), combined (one mask from "
        "the combined density), or per_conformer (mean of per-conformer masks)",
        choices=("off", "combined", "per_conformer"),
    )
    normalize_amplitude: bool = opt(False, help="Score normalized amplitudes (|E|) instead of |F|")
    exclude_free_reflections: bool = opt(False, help="Score the working set only, dropping R-free")
    batch_partition: int = opt(10, help="Ensemble chunk size for the SFcalculator batch")
    sfcalculator_kwargs: dict | None = opt(
        None,
        help="Extra SFcalculator keyword arguments as JSON, e.g. '{\"n_bins\": 15}'",
        json_arg=True,
    )


def option_type(options_cls: type, name: str) -> Any:
    """Return the declared type of one option, with ``None`` stripped from unions.

    ``str | None`` is reported as ``str``: optionality is expressed by the default,
    while consumers (argparse, config coercion) need the underlying value type.
    Non-union hints are returned whole, so ``list[str]`` stays ``list[str]`` rather
    than collapsing to its element type.

    Parameters
    ----------
    options_cls
        A reward's option dataclass.
    name
        Option name.

    Returns
    -------
    Any
        The option's value type, e.g. ``float``, ``bool``, ``list[str]``.
    """
    hint = typing.get_type_hints(options_cls)[name]
    if typing.get_origin(hint) not in (types.UnionType, typing.Union):
        return hint

    args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
    return args[0] if len(args) == 1 else hint


def path_option_names(options_cls: type) -> tuple[str, ...]:
    """Return the options of ``options_cls`` that hold filesystem paths.

    Parameters
    ----------
    options_cls
        A reward's option dataclass.

    Returns
    -------
    tuple[str, ...]
        Names of options declared with ``path=True``.
    """
    return tuple(f.name for f in dataclasses.fields(options_cls) if f.metadata.get("path"))
