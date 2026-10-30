"""Loss configuration: which losses a run uses, with what weights and options.

The configuration is a mapping from loss name to that loss's weight and
options (issue #358)::

    real_space_density:
      weight: 0.4
      loss_options:
        loss_order: 1
    structure_factor:
      weight: 0.6
      loss_options:
        mtzfile: /data/1vme.mtz
        bulk_solvent: combined

Both ways of configuring a run produce this same structure: ``--loss-type``
with per-option flags produces a single entry, and ``--loss-config FILE``
produces one entry per loss in the file. Everything downstream -- building the
losses, serializing the run, injecting per-protein data in a grid search --
works on :class:`LossConfig` and does not care which surface produced it.

Weights are resolved at build time: omitting them all gives every loss ``1/N``,
so a single-loss run is unweighted and a two-loss run is a plain average
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

from sampleworks.core.losses.options import path_option_names
from sampleworks.core.losses.registry import (
    build_single_loss,
    coerce_options,
    get_loss_spec,
    loss_type_names,
    LossBuildContext,
)
from sampleworks.utils.guidance_constants import Losses


if TYPE_CHECKING:
    from sampleworks.core.losses.protocol import LossFunctionProtocol


# Key holding a loss's options inside a configuration file entry.
LOSS_OPTIONS_KEY = "loss_options"
WEIGHT_KEY = "weight"

_YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class LossEntry:
    """One loss in a run: which loss, how strongly, and how configured.

    Attributes
    ----------
    loss
        The loss type.
    weight
        Multiplier on this loss's value in the combined objective. ``None``
        means "unspecified", resolved to ``1/N`` by
        :meth:`LossConfig.resolved_weights`.
    options
        Option values for this loss. Options that are absent take the defaults
        declared in :mod:`sampleworks.core.losses.options`.
    """

    loss: Losses
    weight: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def validated(self) -> LossEntry:
        """Return this entry with its options checked against the loss's schema.

        Returns
        -------
        LossEntry
            The same entry; raises rather than returning on invalid options.

        Raises
        ------
        ValueError
            If the options name something the loss does not have, or the weight
            is negative.
        """
        coerce_options(get_loss_spec(self.loss), self.options)
        if self.weight is not None and self.weight < 0:
            raise ValueError(
                f"Weight for loss '{self.loss.value}' must be non-negative, got {self.weight}."
            )
        return self


@dataclass(frozen=True)
class LossConfig:
    """The full set of losses a guidance run scores against."""

    entries: tuple[LossEntry, ...]

    def __post_init__(self):
        if not self.entries:
            raise ValueError(
                f"A loss configuration needs at least one loss. "
                f"Available loss types: {loss_type_names()}."
            )

        duplicates = sorted({e.loss.value for e in self.entries if self._count(e.loss) > 1})
        if duplicates:
            raise ValueError(f"Loss(s) {duplicates} configured more than once.")

        for entry in self.entries:
            entry.validated()

    def _count(self, loss: Losses) -> int:
        return sum(1 for entry in self.entries if entry.loss is loss)

    @classmethod
    def single(cls, loss: Losses | str, **options: Any) -> LossConfig:
        """Build a one-loss configuration, as ``--loss-type`` produces.

        Parameters
        ----------
        loss
            The loss type to use.
        **options
            Option values for that loss.

        Returns
        -------
        LossConfig
            Configuration holding exactly this loss.
        """
        return cls((LossEntry(loss=Losses(loss), options=dict(options)),))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LossConfig:
        """Build a configuration from the ``{loss: {weight, loss_options}}`` mapping.

        A bare ``{loss: {}}`` (or ``{loss: None}``) is accepted and means
        "this loss, all defaults".

        Parameters
        ----------
        data
            Mapping keyed by loss name.

        Returns
        -------
        LossConfig
            The parsed configuration.

        Raises
        ------
        ValueError
            If a loss name is unknown, an entry is not a mapping, or an entry
            holds keys other than ``weight`` and ``loss_options``.
        """
        entries = []
        for name, raw_entry in data.items():
            loss = get_loss_spec(name).name
            entry = {} if raw_entry is None else raw_entry
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Configuration for loss '{loss.value}' must be a mapping with "
                    f"'{WEIGHT_KEY}' and/or '{LOSS_OPTIONS_KEY}' keys, got "
                    f"{type(entry).__name__}."
                )

            unexpected = sorted(set(entry) - {WEIGHT_KEY, LOSS_OPTIONS_KEY})
            if unexpected:
                raise ValueError(
                    f"Unexpected key(s) {unexpected} in the configuration for loss "
                    f"'{loss.value}'. Loss options belong under '{LOSS_OPTIONS_KEY}'."
                )

            raw_options = entry.get(LOSS_OPTIONS_KEY) or {}
            if not isinstance(raw_options, Mapping):
                raise ValueError(
                    f"'{LOSS_OPTIONS_KEY}' for loss '{loss.value}' must be a mapping of "
                    f"option name to value, got {type(raw_options).__name__}."
                )

            weight = entry.get(WEIGHT_KEY)
            entries.append(
                LossEntry(
                    loss=loss,
                    weight=None if weight is None else float(weight),
                    options=dict(raw_options),
                )
            )

        return cls(tuple(entries))

    @classmethod
    def from_file(cls, path: str | Path) -> LossConfig:
        """Load a loss configuration from a JSON, YAML, or TOML file.

        YAML is read through OmegaConf, so ``${oc.env:VAR}`` interpolation works
        the same way it does in the run presets.

        Parameters
        ----------
        path
            Path to the configuration file; the format follows its suffix.

        Returns
        -------
        LossConfig
            The parsed configuration.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the suffix is not a supported format, or the contents are not a
            mapping of loss names.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Loss configuration file not found: {path}")

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
                f"Unsupported loss configuration format '{suffix or path.name}'. "
                f"Supported formats: {supported}."
            )

        if not isinstance(data, Mapping):
            raise ValueError(
                f"Loss configuration in {path} must be a mapping of loss name to "
                f"{{{WEIGHT_KEY}, {LOSS_OPTIONS_KEY}}}, got {type(data).__name__}."
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
            Mapping keyed by loss name.
        """
        mapping: dict[str, Any] = {}
        for entry in self.entries:
            payload: dict[str, Any] = {}
            if entry.weight is not None:
                payload[WEIGHT_KEY] = entry.weight
            if entry.options:
                payload[LOSS_OPTIONS_KEY] = dict(entry.options)
            mapping[entry.loss.value] = payload
        return mapping

    def with_effective_options(self) -> LossConfig:
        """Return this configuration with every loss's defaults written out.

        A run's metadata should record what actually ran, not only what was typed:
        defaults change between versions, and an option that was defaulted is
        otherwise indistinguishable from one that did not exist. Options left at
        ``None`` stay absent, so they remain fillable by
        :meth:`with_experimental_data`.

        Returns
        -------
        LossConfig
            The same losses, with defaulted option values materialized.
        """
        entries = []
        for entry in self.entries:
            options = coerce_options(get_loss_spec(entry.loss), entry.options)
            effective = {
                name: value
                for name, value in dataclasses.asdict(options).items()
                if value is not None
            }
            entries.append(replace(entry, options=effective))
        return LossConfig(tuple(entries))

    def missing_required_options(self) -> dict[str, tuple[str, ...]]:
        """Report configured losses that are still missing an input they need.

        Lets a caller refuse a configuration before doing expensive work, without
        knowing anything about individual losses. The builders check the same
        thing when they run, which is what protects callers that never come
        through here.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Loss name to the options it is missing, for losses missing any.
        """
        missing = {}
        for entry in self.entries:
            spec = get_loss_spec(entry.loss)
            absent = tuple(
                name for name in spec.required_options if entry.options.get(name) is None
            )
            if absent:
                missing[entry.loss.value] = absent
        return missing

    def resolved_weights(self) -> tuple[float, ...]:
        """Resolve the per-loss weights, filling in the uniform default.

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
            missing = sorted(e.loss.value for e in self.entries if e.weight is None)
            raise ValueError(
                f"Loss(s) {missing} have no weight while others do. Give every loss a "
                f"weight, or none of them (which weights each by 1/{len(self.entries)})."
            )

        weights = tuple(float(entry.weight) for entry in self.entries)  # ty:ignore[invalid-argument-type]
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Loss weights sum to {total:g}, not 1. Using them as given; scale them "
                "yourself if you meant them to be relative."
            )
        return weights

    def with_experimental_data(
        self, *, path: str | Path | None = None, resolution: float | None = None
    ) -> LossConfig:
        """Fill in per-run experimental data without knowing which loss is configured.

        Grid search resolves a map or MTZ and a resolution per protein, long after
        the loss type was chosen. Each loss declares which of its options hold
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
        LossConfig
            A new configuration with the data filled in where it was missing.
        """
        entries = []
        for entry in self.entries:
            spec = get_loss_spec(entry.loss)
            options = dict(entry.options)
            for option_name, value in (
                (spec.data_path_option, None if path is None else str(path)),
                (spec.resolution_option, resolution),
            ):
                if option_name is not None and value is not None:
                    options.setdefault(option_name, value)
            entries.append(replace(entry, options=options))

        return LossConfig(tuple(entries))

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
            options = mapping[entry.loss.value].get(LOSS_OPTIONS_KEY)
            if not options:
                continue
            for option_name in path_option_names(get_loss_spec(entry.loss).options_cls):
                if options.get(option_name) is not None:
                    options[option_name] = remap(str(options[option_name]))
        return mapping


def build_loss(config: LossConfig, context: LossBuildContext) -> LossFunctionProtocol:
    """Build the loss function a run scores against.

    One configured loss at full weight is built and returned directly, so the
    single-loss runs that are today's norm keep exactly the values and gradients
    they had before there was a registry. Anything else becomes a
    :class:`~sampleworks.core.losses.composite.CompositeLoss`.

    Parameters
    ----------
    config
        The run's loss configuration.
    context
        Run-level inputs (the parsed input structure, the device).

    Returns
    -------
    LossFunctionProtocol
        A single loss or a weighted combination of several.
    """
    weights = config.resolved_weights()
    losses = [build_single_loss(entry.loss, entry.options, context) for entry in config.entries]

    if len(losses) == 1 and weights[0] == 1.0:
        return losses[0]

    from sampleworks.core.losses.composite import CompositeLoss

    logger.info(
        "Combining losses: "
        + ", ".join(
            f"{weight:g}*{entry.loss.value}"
            for entry, weight in zip(config.entries, weights, strict=True)
        )
    )
    return CompositeLoss(losses, weights)
