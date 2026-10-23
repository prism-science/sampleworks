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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from sampleworks.core.losses.options import (
    DiffuseBraggOptions,
    RealSpaceDensityOptions,
    StructureFactorOptions,
)
from sampleworks.utils.guidance_constants import Losses


if TYPE_CHECKING:
    import torch
    from sampleworks.core.losses.protocol import LossFunctionProtocol


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
    data_path_option
        Name of the option holding the experimental data file, if any. Lets
        callers that resolve data per protein (grid search) inject it without
        knowing which loss they are configuring.
    resolution_option
        Name of the option holding the resolution, if any. Same rationale.
    required_options
        Options this loss cannot be built without. The builder raises on them
        too -- that is the guarantee for library callers -- but declaring them
        here lets the CLI report a missing one as a usage error, before a model
        is loaded.
    """

    name: Losses
    options_cls: type
    builder_path: str
    description: str
    data_path_option: str | None = None
    resolution_option: str | None = None
    required_options: tuple[str, ...] = ()

    def builder(self) -> Callable[[Any, LossBuildContext], LossFunctionProtocol]:
        """Import and return this loss's builder function.

        Returns
        -------
        Callable
            Builder taking ``(options, context)`` and returning a loss function.
        """
        module_path, function_name = self.builder_path.split(":")
        return getattr(importlib.import_module(module_path), function_name)


LOSS_SPECS: dict[Losses, LossSpec] = {
    Losses.REAL_SPACE_DENSITY: LossSpec(
        name=Losses.REAL_SPACE_DENSITY,
        options_cls=RealSpaceDensityOptions,
        builder_path="sampleworks.core.losses.real_space_density:build_real_space_density_loss",
        description="Real-space density fit (X-ray or cryo-EM map).",
        data_path_option="density",
        resolution_option="resolution",
        required_options=("density", "resolution"),
    ),
    Losses.STRUCTURE_FACTOR: LossSpec(
        name=Losses.STRUCTURE_FACTOR,
        options_cls=StructureFactorOptions,
        builder_path="sampleworks.core.losses.structure_factor:build_structure_factor_loss",
        description="Reciprocal-space structure-factor amplitude fit (MTZ target).",
        data_path_option="mtzfile",
        resolution_option="resolution",
        # No resolution: the MTZ carries its own, and --resolution only truncates it.
        required_options=("mtzfile",),
    ),
    Losses.DIFFUSE_BRAGG: LossSpec(
        name=Losses.DIFFUSE_BRAGG,
        options_cls=DiffuseBraggOptions,
        builder_path="sampleworks.core.losses.diffuse_bragg:build_diffuse_bragg_loss",
        description="Bragg amplitudes plus anisotropic diffuse intensities, via lunus.sf.",
        resolution_option="resolution",
        # Which target is required depends on bragg_weight; the builder checks.
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
