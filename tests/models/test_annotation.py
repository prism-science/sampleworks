"""Tests for the model-annotation dispatcher.

These exercise the dispatch contract (which function, whether ``out_dir`` is passed, kwarg
forwarding) without importing any real wrapper module, by patching the lazy importer. They
therefore run in any environment, with no model dependencies installed.
"""

from __future__ import annotations

import pytest
from sampleworks.models import annotation
from sampleworks.models.annotation import (
    annotate_structure_for_model,
    MODEL_ANNOTATION_REGISTRY,
)
from sampleworks.utils.guidance_constants import StructurePredictor


def test_registry_covers_every_structure_predictor():
    """Every model has a registered annotate function, so dispatch never misses one."""
    assert set(MODEL_ANNOTATION_REGISTRY) == set(StructurePredictor)


def test_unknown_model_type_raises():
    """An unrecognized model type fails fast rather than KeyError-ing on the registry."""
    with pytest.raises(ValueError, match="Unknown model type"):
        annotate_structure_for_model("not_a_model", {})


class _Recorder:
    """Stand-in annotate function that records how it was called and echoes its kwargs."""

    def __init__(self, path: str):
        self.path = path
        self.calls: list[tuple[dict, dict]] = []

    def __call__(self, structure: dict, **kwargs) -> dict:
        self.calls.append((structure, kwargs))
        return {"annotated_by": self.path, **kwargs}


@pytest.fixture
def recorders(monkeypatch):
    """Patch the lazy importer so each annotate-fn path resolves to a recording fake."""
    registry: dict[str, _Recorder] = {}

    def fake_import(path: str) -> _Recorder:
        return registry.setdefault(path, _Recorder(path))

    monkeypatch.setattr(annotation, "_import_from_path", fake_import)
    return registry


@pytest.mark.parametrize(
    "model_type",
    [StructurePredictor.BOLTZ_1, StructurePredictor.BOLTZ_2, StructurePredictor.PROTENIX],
)
def test_out_dir_forwarded_when_supported(model_type, recorders):
    """Models whose annotate function takes out_dir receive it, alongside other kwargs."""
    annotate_structure_for_model(model_type, {"s": 1}, out_dir="/tmp/out", recycling_steps=3)
    path = MODEL_ANNOTATION_REGISTRY[model_type].annotate_fn_path
    ((structure, kwargs),) = recorders[path].calls
    assert structure == {"s": 1}
    assert kwargs == {"out_dir": "/tmp/out", "recycling_steps": 3}


@pytest.mark.parametrize("model_type", [StructurePredictor.RF3, StructurePredictor.PROTPARDELLE])
def test_out_dir_omitted_when_unsupported(model_type, recorders):
    """Models without an out_dir parameter are never handed one, even if a caller passes it."""
    annotate_structure_for_model(model_type, {"s": 1}, out_dir="/tmp/out")
    path = MODEL_ANNOTATION_REGISTRY[model_type].annotate_fn_path
    ((_, kwargs),) = recorders[path].calls
    assert "out_dir" not in kwargs


def test_string_model_type_is_accepted_and_kwargs_forwarded(recorders):
    """A string model type resolves like the enum and model-specific kwargs pass through."""
    result = annotate_structure_for_model("rf3", {"s": 1}, msa_path="x.a3m")
    path = MODEL_ANNOTATION_REGISTRY[StructurePredictor.RF3].annotate_fn_path
    ((_, kwargs),) = recorders[path].calls
    assert kwargs == {"msa_path": "x.a3m"}
    assert result["annotated_by"] == path
