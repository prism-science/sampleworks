"""Registry of loss types, their option schemas, and their builders.

Adding a loss type is: write the loss, declare its options in
:mod:`sampleworks.core.losses.options`, write a ``build_*`` function next to the
loss, and add one :class:`LossSpec` here. The CLI, the configuration-file
schema, and run serialization all follow from that entry -- there is no argparse
plumbing to edit and no dispatch chain to extend.

Builders are addressed by ``"module:function"`` string and imported only when a
loss is actually built, so this module stays importable in environments that
cannot import every loss's dependencies.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from sampleworks.core.losses.options import (
    DiffuseBraggOptions,
    option_type,
    RealSpaceDensityOptions,
    StructureFactorOptions,
)
from sampleworks.utils.guidance_constants import Losses


if TYPE_CHECKING:
    import torch

    from sampleworks.core.losses.protocol import LossFunctionProtocol


_TRUE_TEXT = frozenset({"true", "t", "yes", "y", "on", "1"})
_FALSE_TEXT = frozenset({"false", "f", "no", "n", "off", "0"})


@dataclass(frozen=True)
class LossBuildContext:
    """Run-level inputs a loss builder may need that are not loss options.

    Attributes
    ----------
    structure
        Atomworks-parsed structure dictionary for the input structure, loaded once
        by the caller and shared by every loss in a run.
    device
        Torch device the loss runs on.
    """

    structure: dict
    device: torch.device | str = "cpu"


@dataclass(frozen=True)
class LossSpec:
    """Everything the framework needs to know about a loss without importing it.

    Attributes
    ----------
    name
        The loss's :class:`~sampleworks.utils.guidance_constants.Losses` member,
        which is also its configuration-file key and ``--loss-type`` value.
    options_cls
        Option dataclass declaring this loss's configurable options.
    builder_path
        ``"module:function"`` address of the builder, resolved lazily.
    description
        One-line summary, shown in ``--help``.
    required_options
        Options this loss cannot be built without. The builder raises on them
        too -- that is the guarantee for library callers -- but declaring them
        here lets the CLI and grid search report a missing one before a model
        is loaded.
    conditional_required_options
        For options required only for some values of other options: a function
        from the loss's options to the extra options it needs.
    """

    name: Losses
    options_cls: type
    builder_path: str
    description: str
    required_options: tuple[str, ...] = ()
    conditional_required_options: Callable[[Any], tuple[str, ...]] | None = None

    def required_for(self, options: Any) -> tuple[str, ...]:
        """Return the options this loss needs, given its other option values.

        Parameters
        ----------
        options
            An instance of ``options_cls``.

        Returns
        -------
        tuple[str, ...]
            Names of the options that must not be ``None``.
        """
        if self.conditional_required_options is None:
            return self.required_options
        return self.required_options + self.conditional_required_options(options)

    def builder(self) -> Callable[[Any, LossBuildContext], LossFunctionProtocol]:
        """Import and return this loss's builder function.

        Returns
        -------
        Callable
            Builder taking ``(options, context)`` and returning a loss function.
        """
        module_path, function_name = self.builder_path.split(":")
        return getattr(importlib.import_module(module_path), function_name)


def _diffuse_bragg_required(options: DiffuseBraggOptions) -> tuple[str, ...]:
    """A target is needed only for a term with nonzero weight."""
    required = []
    if options.bragg_weight > 0.0:
        required.append("bragg_target")
    if options.bragg_weight < 1.0:
        required.append("diffuse_target")
    return tuple(required)


LOSS_SPECS: dict[Losses, LossSpec] = {
    Losses.REAL_SPACE_DENSITY: LossSpec(
        name=Losses.REAL_SPACE_DENSITY,
        options_cls=RealSpaceDensityOptions,
        builder_path="sampleworks.core.losses.real_space_density:build_real_space_density_loss",
        description="Real-space density fit (X-ray or cryo-EM map).",
        required_options=("density", "resolution"),
    ),
    Losses.STRUCTURE_FACTOR: LossSpec(
        name=Losses.STRUCTURE_FACTOR,
        options_cls=StructureFactorOptions,
        builder_path="sampleworks.core.losses.structure_factor:build_structure_factor_loss",
        description="Reciprocal-space structure-factor amplitude fit (MTZ target).",
        # No resolution: the MTZ carries its own, and --resolution only truncates it.
        required_options=("mtzfile",),
    ),
    Losses.DIFFUSE_BRAGG: LossSpec(
        name=Losses.DIFFUSE_BRAGG,
        options_cls=DiffuseBraggOptions,
        builder_path="sampleworks.core.losses.diffuse_bragg:build_diffuse_bragg_loss",
        description="Bragg amplitudes plus anisotropic diffuse intensities, via lunus.sf.",
        conditional_required_options=_diffuse_bragg_required,
    ),
}


def get_loss_spec(loss: Losses | str) -> LossSpec:
    """Look up the specification for one loss type.

    Parameters
    ----------
    loss
        A :class:`Losses` member or its string value.

    Returns
    -------
    LossSpec
        The registered specification.

    Raises
    ------
    ValueError
        If ``loss`` is not a registered loss type.
    """
    try:
        return LOSS_SPECS[Losses(loss)]
    except ValueError:
        raise ValueError(
            f"Unknown loss type {str(loss)!r}. Available loss types: {loss_type_names()}."
        ) from None


def loss_type_names() -> list[str]:
    """Return the registered loss type names, for ``choices`` and error messages.

    Returns
    -------
    list[str]
        Loss names in registration order.
    """
    return [spec.name.value for spec in LOSS_SPECS.values()]


def coerce_options(spec: LossSpec, raw: Mapping[str, Any]) -> Any:
    """Validate a raw option mapping and materialize it as the loss's options.

    Parameters
    ----------
    spec
        Specification of the loss the options belong to.
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
        If ``raw`` names an option the loss does not have.
    """
    valid = {f.name for f in dataclasses.fields(spec.options_cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"Unknown option(s) {unknown} for loss '{spec.name.value}'. "
            f"Valid options: {sorted(valid)}."
        )
    return spec.options_cls(**dict(raw))


# Option name -> (option class, field), across every loss. A proteins-CSV column is
# not tied to one loss, so it is parsed by name; losses sharing a name share its
# type (tests/losses/test_loss_registry.py holds them to that), so any one
# declaration serves. Each loss still validates the parsed value when its
# options are built.
_OPTION_DECLARATIONS: dict[str, tuple[type, dataclasses.Field]] = {
    option.name: (spec.options_cls, option)
    for spec in LOSS_SPECS.values()
    for option in dataclasses.fields(spec.options_cls)
}


def loss_option_names() -> list[str]:
    """Return every option name declared by any registered loss.

    Returns
    -------
    list[str]
        Sorted option names.
    """
    return sorted(_OPTION_DECLARATIONS)


def _declaration(name: str) -> tuple[type, dataclasses.Field]:
    try:
        return _OPTION_DECLARATIONS[name]
    except KeyError:
        raise ValueError(
            f"No registered loss has an option named {name!r}. Loss options: {loss_option_names()}."
        ) from None


def is_path_option(name: str) -> bool:
    """Return whether the loss option ``name`` holds a filesystem path.

    Parameters
    ----------
    name
        Option name, as declared by any registered loss.

    Returns
    -------
    bool
        True if the option was declared with ``path=True``.

    Raises
    ------
    ValueError
        If no registered loss declares ``name``.
    """
    _, option = _declaration(name)
    return bool(option.metadata["path"])


def bool_from_text(text: str) -> bool:
    """Parse a boolean written as text, case-insensitively.

    Parameters
    ----------
    text
        One of ``true``/``t``/``yes``/``y``/``on``/``1`` or
        ``false``/``f``/``no``/``n``/``off``/``0``, in any case.

    Returns
    -------
    bool
        The value ``text`` spells.

    Raises
    ------
    ValueError
        If ``text`` is none of those.
    """
    lowered = text.strip().lower()
    if lowered in _TRUE_TEXT:
        return True
    if lowered in _FALSE_TEXT:
        return False
    raise ValueError(f"Expected true or false, got {text!r}.")


def option_from_text(name: str, text: str) -> Any:
    """Parse a loss option's value from text, such as a cell of a proteins CSV.

    Parameters
    ----------
    name
        Option name, as declared by any registered loss.
    text
        The value as written: see :func:`bool_from_text` for booleans,
        whitespace-separated items for lists, JSON for dict-valued options.

    Returns
    -------
    Any
        The value converted to the option's declared type.

    Raises
    ------
    ValueError
        If no registered loss declares ``name``, or ``text`` is not a valid
        value for it.
    """
    options_cls, option = _declaration(name)
    value_type = option_type(options_cls, name)
    if option.metadata["json_arg"]:
        return json.loads(text)
    if value_type is bool:
        return bool_from_text(text)
    # value_type is a type hint such as list[str], not a list, so isinstance does not
    # apply; add_loss_args unpacks list options the same way.
    if typing.get_origin(value_type) is list:
        item_type = typing.get_args(value_type)[0]
        return [item_type(item) for item in text.split()]
    return value_type(text)


def build_single_loss(
    loss: Losses | str,
    options: Mapping[str, Any],
    context: LossBuildContext,
) -> LossFunctionProtocol:
    """Build one loss from its option mapping.

    Parameters
    ----------
    loss
        Loss type to build.
    options
        Option values for that loss; missing options take their defaults.
    context
        Run-level inputs (parsed structure, device).

    Returns
    -------
    LossFunctionProtocol
        The constructed loss function.
    """
    spec = get_loss_spec(loss)
    return spec.builder()(coerce_options(spec, options), context)
