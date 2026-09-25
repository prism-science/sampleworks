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
rewards, serializing the run, injecting per-protein data in a grid search --
works on :class:`RewardConfig` and does not care which surface produced it.

Weights are resolved at build time: omitting them all gives every reward ``1/N``,
so a single-reward run is unweighted and a two-reward run is a plain average
unless the user says otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from collections.abc import Mapping
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
    RewardBuildContext,
)
from sampleworks.utils.guidance_constants import Rewards


if TYPE_CHECKING:
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
        Multiplier on this reward's value in the combined objective. ``None``
        means "unspecified", resolved to ``1/N`` by
        :meth:`RewardConfig.resolved_weights`.
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
            If a reward name is unknown, an entry is not a mapping, or an entry
            holds keys other than ``weight`` and ``reward_options``.
        """
        entries = []
        for name, raw_entry in data.items():
            reward = get_reward_spec(name).name
            entry = {} if raw_entry is None else raw_entry
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Configuration for reward '{reward.value}' must be a mapping with "
                    f"'{WEIGHT_KEY}' and/or '{REWARD_OPTIONS_KEY}' keys, got "
                    f"{type(entry).__name__}."
                )

            unexpected = sorted(set(entry) - {WEIGHT_KEY, REWARD_OPTIONS_KEY})
            if unexpected:
                raise ValueError(
                    f"Unexpected key(s) {unexpected} in the configuration for reward "
                    f"'{reward.value}'. Reward options belong under '{REWARD_OPTIONS_KEY}'."
                )

            raw_options = entry.get(REWARD_OPTIONS_KEY) or {}
            if not isinstance(raw_options, Mapping):
                raise ValueError(
                    f"'{REWARD_OPTIONS_KEY}' for reward '{reward.value}' must be a mapping of "
                    f"option name to value, got {type(raw_options).__name__}."
                )

            weight = entry.get(WEIGHT_KEY)
            entries.append(
                RewardEntry(
                    reward=reward,
                    weight=None if weight is None else float(weight),
                    options=dict(raw_options),
                )
            )

        return cls(tuple(entries))

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

        if not isinstance(data, Mapping):
            raise ValueError(
                f"Reward configuration in {path} must be a mapping of reward name to "
                f"{{{WEIGHT_KEY}, {REWARD_OPTIONS_KEY}}}, got {type(data).__name__}."
            )

        return cls.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        """Return the configuration as the plain mapping it was parsed from.

        Round-trips through :meth:`from_mapping`. Values are primitives only, so
        the result is safe to JSON-encode and to pickle across sampleworks
        versions.

        Returns
        -------
        dict[str, Any]
            Mapping keyed by reward name.
        """
        mapping: dict[str, Any] = {}
        for entry in self.entries:
            payload: dict[str, Any] = {}
            if entry.weight is not None:
                payload[WEIGHT_KEY] = entry.weight
            if entry.options:
                payload[REWARD_OPTIONS_KEY] = dict(entry.options)
            mapping[entry.reward.value] = payload
        return mapping

    def with_effective_options(self) -> RewardConfig:
        """Return this configuration with every reward's defaults written out.

        A run's metadata should record what actually ran, not only what was typed:
        defaults change between versions, and an option that was defaulted is
        otherwise indistinguishable from one that did not exist. Options left at
        ``None`` stay absent, so they remain fillable by
        :meth:`with_experimental_data`.

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
        """Resolve the per-reward weights, filling in the uniform default.

        Returns
        -------
        tuple[float, ...]
            One weight per entry, in order.

        Raises
        ------
        ValueError
            If some but not all entries carry a weight -- the uniform default
            would silently disagree with the weights that were given.
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

        weights = tuple(float(entry.weight) for entry in self.entries)  # ty:ignore[invalid-argument-type]
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Reward weights sum to {total:g}, not 1. Using them as given; scale them "
                "yourself if you meant them to be relative."
            )
        return weights

    def with_experimental_data(
        self, *, path: str | Path | None = None, resolution: float | None = None
    ) -> RewardConfig:
        """Fill in per-run experimental data without knowing which reward is configured.

        Grid search resolves a map or MTZ and a resolution per protein, long after
        the reward type was chosen. Each reward declares which of its options hold
        those (``data_path_option`` / ``resolution_option``), so they can be
        injected generically. Options already set are left alone -- an explicit
        value from the user or a config file wins.

        Parameters
        ----------
        path
            Experimental data file for this run (map or MTZ).
        resolution
            Resolution in Angstroms for this run.

        Returns
        -------
        RewardConfig
            A new configuration with the data filled in where it was missing.
        """
        entries = []
        for entry in self.entries:
            spec = get_reward_spec(entry.reward)
            options = dict(entry.options)
            for option_name, value in (
                (spec.data_path_option, None if path is None else str(path)),
                (spec.resolution_option, resolution),
            ):
                if option_name is not None and value is not None:
                    options.setdefault(option_name, value)
            entries.append(replace(entry, options=options))

        return RewardConfig(tuple(entries))

    def remapped_paths(self, remap: Any) -> dict[str, Any]:
        """Return :meth:`to_mapping` with path-valued options passed through ``remap``.

        Used when writing run metadata, so a run executed in a container records
        host paths like every other path in the configuration.

        Parameters
        ----------
        remap
            Callable taking a path string and returning the path to record.

        Returns
        -------
        dict[str, Any]
            The configuration mapping, with path options remapped.
        """
        mapping = self.to_mapping()
        for entry in self.entries:
            options = mapping[entry.reward.value].get(REWARD_OPTIONS_KEY)
            if not options:
                continue
            for option_name in path_option_names(get_reward_spec(entry.reward).options_cls):
                if options.get(option_name) is not None:
                    options[option_name] = remap(str(options[option_name]))
        return mapping


def build_reward(config: RewardConfig, context: RewardBuildContext) -> RewardFunctionProtocol:
    """Build the reward function a run scores against.

    One configured reward at full weight is built and returned directly, so the
    single-reward runs that are today's norm keep exactly the values and gradients
    they had before there was a registry. Anything else becomes a
    :class:`~sampleworks.core.rewards.composite.CompositeReward`.

    Parameters
    ----------
    config
        The run's reward configuration.
    context
        Run-level inputs (the parsed input structure, the device).

    Returns
    -------
    RewardFunctionProtocol
        A single reward or a weighted combination of several.
    """
    weights = config.resolved_weights()
    rewards = [
        build_single_reward(entry.reward, entry.options, context) for entry in config.entries
    ]

    if len(rewards) == 1 and weights[0] == 1.0:
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
