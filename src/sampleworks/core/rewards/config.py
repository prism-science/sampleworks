"""Reward configuration: which rewards a run uses, with what weights and options.

The configuration is a mapping from reward name to that reward's weight and
options (issue #358)::

    real_space_density:
      weight: 0.4
      reward_options:
        loss_order: 1
    structure_factor:
      weight: 0.6
      reward_options:
        mtzfile: /data/1vme.mtz
        bulk_solvent: combined

Both ways of configuring a run produce this same structure: ``--reward-type``
with per-option flags produces a single entry, and ``--reward-config FILE``
produces one entry per reward in the file. Everything downstream -- building the
rewards, serializing the run -- works on :class:`RewardConfig` and does not care
which surface produced it.

Weights are relative and resolved at build time, normalized to sum to 1: omitting
them all gives every reward ``1/N``, ``{a: 2, b: 3}`` gives ``(0.4, 0.6)``, and a
single reward always has weight 1. The overall strength of guidance is the
scaler's step size, so the ratios written here are the only thing that matters.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger
from sampleworks.core.rewards.options import path_option_names
from sampleworks.core.rewards.registry import (
    build_single_reward,
    coerce_options,
    get_reward_spec,
    reward_type_names,
)
from sampleworks.utils.guidance_constants import Rewards


if TYPE_CHECKING:
    import torch
    from sampleworks.core.rewards.protocol import RewardFunctionProtocol


# Key holding a reward's options inside a configuration file entry.
REWARD_OPTIONS_KEY = "reward_options"
WEIGHT_KEY = "weight"

_YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class RewardEntry:
    """One reward in a run: which reward, how strongly, and how configured.

    Attributes
    ----------
    reward
        The reward type.
    weight
        Relative weight of this reward in the combined objective; only the ratios
        between entries matter, see :meth:`RewardConfig.resolved_weights`.
        ``None`` means "unspecified".
    options
        Option values for this reward. Options that are absent take the defaults
        declared in :mod:`sampleworks.core.rewards.options`.
    """

    reward: Rewards
    weight: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def validated(self) -> RewardEntry:
        """Return this entry with its options checked against the reward's schema.

        Returns
        -------
        RewardEntry
            The same entry; raises rather than returning on invalid options.

        Raises
        ------
        ValueError
            If the options name something the reward does not have, or the weight
            is negative.
        """
        coerce_options(get_reward_spec(self.reward), self.options)
        if self.weight is not None and self.weight < 0:
            raise ValueError(
                f"Weight for reward '{self.reward.value}' must be non-negative, got {self.weight}."
            )
        return self

    @classmethod
    def from_mapping(cls, reward: Rewards | str, entry: Mapping[str, Any] | None) -> RewardEntry:
        """Parse one ``{weight, reward_options}`` entry of a configuration file.

        A bare ``{}`` (or ``None``) means "this reward, all defaults". This checks
        the entry's shape; whether its options fit the reward's schema is
        :meth:`validated`, which :class:`RewardConfig` runs on every entry it holds.

        Parameters
        ----------
        reward
            The reward the entry configures, as a member or its name.
        entry
            The entry's value in the ``{reward: {...}}`` mapping.

        Returns
        -------
        RewardEntry
            The parsed entry.

        Raises
        ------
        ValueError
            If the reward is unknown, the entry or its ``reward_options`` is not
            a mapping, or the entry holds keys other than ``weight`` and
            ``reward_options``.
        """
        reward = get_reward_spec(reward).name
        entry = _as_mapping(
            {} if entry is None else entry,
            f"Configuration for reward '{reward.value}'",
            f"with '{WEIGHT_KEY}' and/or '{REWARD_OPTIONS_KEY}' keys",
        )

        unexpected = sorted(set(entry) - {WEIGHT_KEY, REWARD_OPTIONS_KEY})
        if unexpected:
            raise ValueError(
                f"Unexpected key(s) {unexpected} in the configuration for reward "
                f"'{reward.value}'. Reward options belong under '{REWARD_OPTIONS_KEY}'."
            )

        options = _as_mapping(
            entry.get(REWARD_OPTIONS_KEY) or {},
            f"'{REWARD_OPTIONS_KEY}' for reward '{reward.value}'",
            "of option name to value",
        )
        weight = entry.get(WEIGHT_KEY)
        return cls(
            reward=reward,
            weight=None if weight is None else float(weight),
            options=dict(options),
        )

    def to_mapping(self, remap_path: Callable[[str], str] | None = None) -> dict[str, Any]:
        """Return this entry as the ``{weight, reward_options}`` mapping it came from.

        Parameters
        ----------
        remap_path
            If given, every path-valued option is passed through it; see
            :meth:`RewardConfig.to_mapping`.

        Returns
        -------
        dict[str, Any]
            The entry, without keys for an unset weight or empty options.
        """
        payload: dict[str, Any] = {}
        if self.weight is not None:
            payload[WEIGHT_KEY] = self.weight
        if self.options:
            options = dict(self.options)
            if remap_path is not None:
                for name in path_option_names(get_reward_spec(self.reward).options_cls):
                    if options.get(name) is not None:
                        options[name] = remap_path(str(options[name]))
            payload[REWARD_OPTIONS_KEY] = options
        return payload


def _as_mapping(value: Any, what: str, shape: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a mapping, else raise naming what should have been one."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a mapping {shape}, got {type(value)}.")
    return value


@dataclass(frozen=True)
class RewardConfig:
    """The full set of rewards a guidance run scores against."""

    entries: tuple[RewardEntry, ...]

    def __post_init__(self):
        if not self.entries:
            raise ValueError(
                f"A reward configuration needs at least one reward. "
                f"Available reward types: {reward_type_names()}."
            )

        duplicates = sorted({e.reward.value for e in self.entries if self._count(e.reward) > 1})
        if duplicates:
            raise ValueError(f"Reward(s) {duplicates} configured more than once.")

        for entry in self.entries:
            entry.validated()

    def _count(self, reward: Rewards) -> int:
        return sum(1 for entry in self.entries if entry.reward is reward)

    @classmethod
    def single(cls, reward: Rewards | str, **options: Any) -> RewardConfig:
        """Build a one-reward configuration, as ``--reward-type`` produces.

        Parameters
        ----------
        reward
            The reward type to use.
        **options
            Option values for that reward.

        Returns
        -------
        RewardConfig
            Configuration holding exactly this reward.
        """
        return cls((RewardEntry(reward=Rewards(reward), options=dict(options)),))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RewardConfig:
        """Build a configuration from the ``{reward: {weight, reward_options}}`` mapping.

        A bare ``{reward: {}}`` (or ``{reward: None}``) is accepted and means
        "this reward, all defaults".

        Parameters
        ----------
        data
            Mapping keyed by reward name.

        Returns
        -------
        RewardConfig
            The parsed configuration.

        Raises
        ------
        ValueError
            If an entry is malformed (see :meth:`RewardEntry.from_mapping`) or
            its options do not fit the reward (see :meth:`RewardEntry.validated`).
        """
        return cls(tuple(RewardEntry.from_mapping(name, entry) for name, entry in data.items()))

    @classmethod
    def from_file(cls, path: str | Path) -> RewardConfig:
        """Load a reward configuration from a JSON, YAML, or TOML file.

        YAML is read through OmegaConf, so ``${oc.env:VAR}`` interpolation works
        the same way it does in the run presets.

        Parameters
        ----------
        path
            Path to the configuration file; the format follows its suffix.

        Returns
        -------
        RewardConfig
            The parsed configuration.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the suffix is not a supported format, or the contents are not a
            mapping of reward names.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Reward configuration file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
        elif suffix in _YAML_SUFFIXES:
            from omegaconf import OmegaConf

            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        elif suffix == ".toml":
            data = tomllib.loads(path.read_text())
        else:
            supported = ", ".join((".json", *_YAML_SUFFIXES, ".toml"))
            raise ValueError(
                f"Unsupported reward configuration format '{suffix or path.name}'. "
                f"Supported formats: {supported}."
            )

        return cls.from_mapping(
            _as_mapping(
                data,
                f"Reward configuration in {path}",
                f"of reward name to {{{WEIGHT_KEY}, {REWARD_OPTIONS_KEY}}}",
            )
        )

    def to_mapping(self, remap_path: Callable[[str], str] | None = None) -> dict[str, Any]:
        """Return the configuration as the plain mapping it was parsed from.

        Round-trips through :meth:`from_mapping`. Values are primitives only, so
        the result is safe to JSON-encode and to pickle across sampleworks
        versions.

        Parameters
        ----------
        remap_path
            If given, every path-valued option (those declared with ``path=True``)
            is passed through it first. Run metadata uses this with the same
            container-to-host remapping every other recorded path goes through,
            so a run executed in a container records host paths.

        Returns
        -------
        dict[str, Any]
            Mapping keyed by reward name.
        """
        return {entry.reward.value: entry.to_mapping(remap_path) for entry in self.entries}

    def with_effective_options(self) -> RewardConfig:
        """Return this configuration with every reward's defaults written out.

        A run's metadata should record what actually ran, not only what was typed:
        defaults change between versions, and an option that was defaulted is
        otherwise indistinguishable from one that did not exist. Options whose
        value is ``None`` stay absent, so the mapping records only what is set.

        Returns
        -------
        RewardConfig
            The same rewards, with defaulted option values materialized.
        """
        entries = []
        for entry in self.entries:
            options = coerce_options(get_reward_spec(entry.reward), entry.options)
            effective = {
                name: value
                for name, value in dataclasses.asdict(options).items()
                if value is not None
            }
            entries.append(replace(entry, options=effective))
        return RewardConfig(tuple(entries))

    def missing_required_options(self) -> dict[str, tuple[str, ...]]:
        """Report configured rewards that are still missing an input they need.

        Lets a caller refuse a configuration before doing expensive work, without
        knowing anything about individual rewards. The builders check the same
        thing when they run, which is what protects callers that never come
        through here.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Reward name to the options it is missing, for rewards missing any.
        """
        missing = {}
        for entry in self.entries:
            spec = get_reward_spec(entry.reward)
            absent = tuple(
                name for name in spec.required_options if entry.options.get(name) is None
            )
            if absent:
                missing[entry.reward.value] = absent
        return missing

    def resolved_weights(self) -> tuple[float, ...]:
        """Resolve the per-reward weights, normalized to sum to 1.

        Weights are relative: ``{a: 2, b: 3}`` and ``{a: 0.4, b: 0.6}`` are the
        same configuration, and a single reward has weight 1 whatever was written.
        The overall strength of guidance is the scaler's step size, not these.

        Returns
        -------
        tuple[float, ...]
            One weight per entry, in order, summing to 1.

        Raises
        ------
        ValueError
            If some but not all entries carry a weight -- the uniform default
            would silently disagree with the weights that were given -- or if
            every given weight is zero.
        """
        weighted = [entry for entry in self.entries if entry.weight is not None]
        if not weighted:
            return tuple(1.0 / len(self.entries) for _ in self.entries)

        if len(weighted) != len(self.entries):
            missing = sorted(e.reward.value for e in self.entries if e.weight is None)
            raise ValueError(
                f"Reward(s) {missing} have no weight while others do. Give every reward a "
                f"weight, or none of them (which weights each by 1/{len(self.entries)})."
            )

        weights = [float(entry.weight) for entry in self.entries]  # ty:ignore[invalid-argument-type]
        total = sum(weights)
        if total <= 0:
            raise ValueError("Reward weights are all zero; at least one must be positive.")
        return tuple(weight / total for weight in weights)


def build_reward(
    config: RewardConfig, *, device: torch.device | str = "cpu"
) -> RewardFunctionProtocol:
    """Build the reward function a run scores against.

    One configured reward is built and returned directly -- its normalized weight
    is necessarily 1 -- so the single-reward runs that are today's norm keep
    exactly the values and gradients they had before there was a registry. Several
    rewards become a :class:`~sampleworks.core.rewards.composite.CompositeReward`
    with their normalized weights.

    Parameters
    ----------
    config
        The run's reward configuration.
    device
        Torch device the rewards run on. Nothing else is needed at build time:
        a reward that depends on the model topology binds to it later, in
        :meth:`~sampleworks.core.rewards.protocol.PreparableRewardFunctionProtocol.prepare`.

    Returns
    -------
    RewardFunctionProtocol
        A single reward or a weighted combination of several.
    """
    weights = config.resolved_weights()
    rewards = [
        build_single_reward(entry.reward, entry.options, device=device) for entry in config.entries
    ]

    if len(rewards) == 1:
        return rewards[0]

    from sampleworks.core.rewards.composite import CompositeReward

    logger.info(
        "Combining rewards: "
        + ", ".join(
            f"{weight:g}*{entry.reward.value}"
            for entry, weight in zip(config.entries, weights, strict=True)
        )
    )
    return CompositeReward(rewards, weights)
