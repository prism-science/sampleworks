"""Dispatch a structure to the right model-specific annotation function.

Each model wrapper exposes an annotate function that stamps an atomworks structure
with the configuration that model needs before inference
(:func:`~sampleworks.models.boltz.wrapper.process_structure_for_boltz`,
:func:`~sampleworks.models.protenix.wrapper.annotate_structure_for_protenix`, and so
on). Callers that work across models -- the guidance runner, the test suite -- otherwise
repeat the same ``if model is Boltz ... elif model is RF3 ...`` dispatch to pick and import
the right one.

This module centralizes that dispatch. It maps each
:class:`~sampleworks.utils.guidance_constants.StructurePredictor` to its annotate function
and imports the function lazily, so importing this module never pulls in any model's
dependencies -- which matters because the wrappers' dependencies are mutually incompatible
and installed in separate environments.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sampleworks.utils.guidance_constants import StructurePredictor


@dataclass(frozen=True)
class AnnotationSpec:
    """How to call one model's annotate function.

    Parameters
    ----------
    annotate_fn_path
        Fully qualified path to the annotate function, imported lazily.
    requires_out_dir
        Whether the annotate function accepts an ``out_dir`` keyword. Functions
        that write intermediate files (Boltz, Protenix) take one; the others do not,
        so passing ``out_dir`` to them would raise :class:`TypeError`.
    """

    annotate_fn_path: str
    requires_out_dir: bool


# Source of truth for which annotate function belongs to which model. Boltz 1 and 2 share
# one function. Keep this in sync with the wrapper modules.
MODEL_ANNOTATION_REGISTRY: dict[StructurePredictor, AnnotationSpec] = {
    StructurePredictor.BOLTZ_1: AnnotationSpec(
        "sampleworks.models.boltz.wrapper.process_structure_for_boltz", requires_out_dir=True
    ),
    StructurePredictor.BOLTZ_2: AnnotationSpec(
        "sampleworks.models.boltz.wrapper.process_structure_for_boltz", requires_out_dir=True
    ),
    StructurePredictor.PROTENIX: AnnotationSpec(
        "sampleworks.models.protenix.wrapper.annotate_structure_for_protenix",
        requires_out_dir=True,
    ),
    StructurePredictor.RF3: AnnotationSpec(
        "sampleworks.models.rf3.wrapper.annotate_structure_for_rf3", requires_out_dir=False
    ),
    StructurePredictor.PROTPARDELLE: AnnotationSpec(
        "sampleworks.models.protpardelle.wrapper.annotate_structure_for_protpardelle",
        requires_out_dir=False,
    ),
}


def _import_from_path(path: str) -> Any:
    """Import and return the object named by a fully qualified dotted path.

    Parameters
    ----------
    path
        Dotted path such as ``"sampleworks.models.rf3.wrapper.annotate_structure_for_rf3"``.

    Returns
    -------
    Any
        The imported object.
    """
    module_path, obj_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), obj_name)


def annotate_structure_for_model(
    model_type: StructurePredictor | str,
    structure: dict,
    *,
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict:
    """Annotate a structure with a model's inference configuration.

    Resolves the annotate function for *model_type*, imports it lazily, and calls it. The
    model-specific keyword arguments (``recycling_steps``, ``msa_path``, ...) are the
    caller's responsibility and are forwarded unchanged; this function only owns the
    dispatch, the import, and whether ``out_dir`` is passed.

    Parameters
    ----------
    model_type
        Which model to annotate for, as a :class:`StructurePredictor` or its string value.
    structure
        Atomworks structure dictionary to annotate.
    out_dir
        Output directory for models whose annotate function writes intermediate files
        (Boltz, Protenix). Ignored for models that do not accept one.
    **kwargs
        Model-specific keyword arguments forwarded to the annotate function. Passing an
        argument the target function does not accept raises :class:`TypeError`.

    Returns
    -------
    dict
        The annotated structure.

    Raises
    ------
    ValueError
        If *model_type* is not a known :class:`StructurePredictor`.
    """
    try:
        model_type = StructurePredictor(model_type)
    except ValueError as error:
        known = [m.value for m in MODEL_ANNOTATION_REGISTRY]
        raise ValueError(
            f"Unknown model type for structure annotation: {model_type!r}. Expected one of {known}."
        ) from error

    spec = MODEL_ANNOTATION_REGISTRY[model_type]
    annotate_fn = _import_from_path(spec.annotate_fn_path)
    if spec.requires_out_dir:
        return annotate_fn(structure, out_dir=out_dir, **kwargs)
    return annotate_fn(structure, **kwargs)
