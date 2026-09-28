"""Registry of reward types, their option schemas, and their builders.

Adding a reward type is: write the reward, declare its options in
:mod:`sampleworks.core.rewards.options`, write a ``build_*`` function next to the
reward, and add one :class:`RewardSpec` here. The CLI, the configuration-file
schema, and run serialization all follow from that entry -- there is no argparse
plumbing to edit and no dispatch chain to extend.

Builders are addressed by ``"module:function"`` string and imported only when a
reward is actually built, so this module stays importable in environments that
cannot import every reward's dependencies.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from sampleworks.core.rewards.options import (
    RealSpaceDensityOptions,
    StructureFactorOptions,
)
from sampleworks.utils.guidance_constants import Rewards


if TYPE_CHECKING:
    import torch
    from sampleworks.core.rewards.protocol import RewardFunctionProtocol


@dataclass(frozen=True)
class RewardBuildContext:
    """Run-level inputs a reward builder may need that are not reward options.

    Attributes
    ----------
    structure
        Atomworks-parsed structure dictionary for the input structure, loaded once
        by the caller and shared by every reward in a run.
    device
        Torch device the reward runs on.
    """

    structure: dict
    device: torch.device | str = "cpu"


@dataclass(frozen=True)
class RewardSpec:
    """Everything the framework needs to know about a reward without importing it.

    Attributes
    ----------
    name
        The reward's :class:`~sampleworks.utils.guidance_constants.Rewards` member,
        which is also its configuration-file key and ``--reward-type`` value.
    options_cls
        Option dataclass declaring this reward's configurable options.
    builder_path
        ``"module:function"`` address of the builder, resolved lazily.
    description
        One-line summary, shown in ``--help``.
    data_path_option
        Name of the option holding the experimental data file, if any. Lets
        callers that resolve data per protein (grid search) inject it without
        knowing which reward they are configuring.
    resolution_option
        Name of the option holding the resolution, if any. Same rationale.
    required_options
        Options this reward cannot be built without. The builder raises on them
        too -- that is the guarantee for library callers -- but declaring them
        here lets the CLI report a missing one as a usage error, before a model
        is loaded.
    """

    name: Rewards
    options_cls: type
    builder_path: str
    description: str
    data_path_option: str | None = None
    resolution_option: str | None = None
    required_options: tuple[str, ...] = ()

    def builder(self) -> Callable[[Any, RewardBuildContext], RewardFunctionProtocol]:
        """Import and return this reward's builder function.

        Returns
        -------
        Callable
            Builder taking ``(options, context)`` and returning a reward function.
        """
        module_path, function_name = self.builder_path.split(":")
        return getattr(importlib.import_module(module_path), function_name)


REWARD_SPECS: dict[Rewards, RewardSpec] = {
    Rewards.REAL_SPACE_DENSITY: RewardSpec(
        name=Rewards.REAL_SPACE_DENSITY,
        options_cls=RealSpaceDensityOptions,
        builder_path="sampleworks.core.rewards.real_space_density:build_real_space_density_reward",
        description="Real-space density fit (X-ray or cryo-EM map).",
        data_path_option="density",
        resolution_option="resolution",
        required_options=("density", "resolution"),
    ),
    Rewards.STRUCTURE_FACTOR: RewardSpec(
        name=Rewards.STRUCTURE_FACTOR,
        options_cls=StructureFactorOptions,
        builder_path="sampleworks.core.rewards.structure_factor:build_structure_factor_reward",
        description="Reciprocal-space structure-factor amplitude fit (MTZ target).",
        data_path_option="mtzfile",
        resolution_option="resolution",
        # No resolution: the MTZ carries its own, and --resolution only truncates it.
        required_options=("mtzfile",),
    ),
}


def get_reward_spec(reward: Rewards | str) -> RewardSpec:
    """Look up the specification for one reward type.

    Parameters
    ----------
    reward
        A :class:`Rewards` member or its string value.

    Returns
    -------
    RewardSpec
        The registered specification.

    Raises
    ------
    ValueError
        If ``reward`` is not a registered reward type.
    """
    try:
        return REWARD_SPECS[Rewards(reward)]
    except ValueError:
        raise ValueError(
            f"Unknown reward type {str(reward)!r}. Available reward types: {reward_type_names()}."
        ) from None


def reward_type_names() -> list[str]:
    """Return the registered reward type names, for ``choices`` and error messages.

    Returns
    -------
    list[str]
        Reward names in registration order.
    """
    return [spec.name.value for spec in REWARD_SPECS.values()]


def coerce_options(spec: RewardSpec, raw: Mapping[str, Any]) -> Any:
    """Validate a raw option mapping and materialize it as the reward's options.

    Parameters
    ----------
    spec
        Specification of the reward the options belong to.
    raw
        Option values from a CLI namespace or a configuration file. Options that
        are absent take the dataclass default.

    Returns
    -------
    Any
        An instance of ``spec.options_cls``.

    Raises
    ------
    ValueError
        If ``raw`` names an option the reward does not have.
    """
    valid = {f.name for f in dataclasses.fields(spec.options_cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"Unknown option(s) {unknown} for reward '{spec.name.value}'. "
            f"Valid options: {sorted(valid)}."
        )
    return spec.options_cls(**dict(raw))


def build_single_reward(
    reward: Rewards | str,
    options: Mapping[str, Any],
    context: RewardBuildContext,
) -> RewardFunctionProtocol:
    """Build one reward from its option mapping.

    Parameters
    ----------
    reward
        Reward type to build.
    options
        Option values for that reward; missing options take their defaults.
    context
        Run-level inputs (parsed structure, device).

    Returns
    -------
    RewardFunctionProtocol
        The constructed reward function.
    """
    spec = get_reward_spec(reward)
    return spec.builder()(coerce_options(spec, options), context)
