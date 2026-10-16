"""Selection and stripping on multi-model input.

Per-atom annotations are shared across models, so one mask filters every model.
The two options a stack cannot express, per-model occupancies and altloc
expansion, are refused.
"""

from __future__ import annotations

import numpy as np
import pytest
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx import CIFFile, set_structure


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

from pathlib import Path

from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations
from sampleworks.synthetic.synthetic_utils import BatchRowForMTZ


@pytest.fixture(scope="module")
def source_1vme(resources_dir: Path) -> Path:
    source = resources_dir / "1vme" / "1vme_final.cif"
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")
    return source


@pytest.fixture(scope="module")
def multi_model_cif(source_1vme: Path, tmp_path_factory) -> Path:
    """A two-model CIF written from 1VME's altloc conformations; the repo ships none."""
    expanded = load_configurations(
        source_1vme, BatchRowForMTZ(filename=source_1vme.name), "default", altlocs_as_models=True
    )
    assert expanded.stack_depth() >= 2, "1VME should expand to at least two conformations"

    stack = AtomArrayStack(expanded.stack_depth(), expanded.array_length())
    for field in ("chain_id", "res_id", "res_name", "atom_name", "element", "hetero"):
        stack.set_annotation(field, expanded.get_annotation(field))
    stack.coord = expanded.coord.astype(np.float32)

    path = tmp_path_factory.mktemp("multi_model") / "ensemble.cif"
    cif = CIFFile()
    set_structure(cif, stack)
    cif.write(path)
    return path


@pytest.fixture(params=["multi_model", "altlocs_as_models"])
def ensemble_input(request, multi_model_cif: Path, source_1vme: Path):
    """The two ways an ensemble reaches load_configurations: a multi-model file, or a
    single-model file expanded by altloc. Returns ``(path, extra kwargs)``."""
    if request.param == "multi_model":
        return multi_model_cif, {}
    return source_1vme, {"altlocs_as_models": True}


def _load(path: Path, **kwargs):
    return load_configurations(
        path, BatchRowForMTZ(filename=path.name, **kwargs.pop("row", {})), "default", **kwargs
    )


def test_every_configuration_survives_the_load(ensemble_input):
    path, kwargs = ensemble_input
    assert _load(path, **kwargs).stack_depth() >= 2


def test_selection_applies_to_every_configuration(ensemble_input):
    """Selection drops atoms from every model and keeps every model, on both paths."""
    path, kwargs = ensemble_input
    full = _load(path, **kwargs)
    selected = _load(path, row={"selection": "chain A"}, **kwargs)

    assert selected.stack_depth() == full.stack_depth(), "selection must not drop models"
    assert selected.array_length() < full.array_length(), "selection must drop atoms"
    assert set(np.asarray(selected.chain_id)) == {"A"}


def test_stripping_hydrogens_applies_to_every_configuration(ensemble_input):
    path, kwargs = ensemble_input
    stripped = _load(path, strip_hydrogens=True, **kwargs)

    assert "H" not in set(np.asarray(stripped.element))
    assert stripped.stack_depth() >= 2


def test_stripping_waters_applies_to_every_configuration(ensemble_input):
    path, kwargs = ensemble_input
    full = _load(path, **kwargs)
    stripped = _load(path, strip_waters=True, **kwargs)

    assert "HOH" not in set(np.asarray(stripped.res_name))
    assert stripped.stack_depth() == full.stack_depth()


def test_topology_and_coordinates_stay_in_register_after_filtering(ensemble_input):
    """The shared annotations describe every model's coordinates."""
    path, kwargs = ensemble_input
    selected = _load(path, row={"selection": "chain A"}, **kwargs)

    assert selected.coord.shape == (selected.stack_depth(), selected.array_length(), 3)
    assert len(selected.element) == selected.array_length()


def test_occupancy_options_are_refused_for_multi_model_input(multi_model_cif: Path):
    """A stack has one shared occupancy annotation, so per-model occupancies are refused."""
    with pytest.raises(ValueError, match="occupancy-mode"):
        load_configurations(
            multi_model_cif, BatchRowForMTZ(filename=multi_model_cif.name), "custom"
        )
    with pytest.raises(ValueError, match="occupancy-values"):
        _load(multi_model_cif, row={"occupancy_values": [0.5, 0.5]})


def test_occupancy_options_are_refused_for_altloc_expansion(source_1vme: Path):
    """The expansion sets alternate atoms to full occupancy and weights configurations
    equally, so a requested occupancy has nowhere to go."""
    with pytest.raises(ValueError, match="not supported with --altlocs-as-models"):
        load_configurations(
            source_1vme,
            BatchRowForMTZ(filename=source_1vme.name),
            "uniform",
            altlocs_as_models=True,
        )
    with pytest.raises(ValueError, match="not supported with --altlocs-as-models"):
        _load(source_1vme, row={"occupancy_values": [0.5, 0.5]}, altlocs_as_models=True)


def test_altlocs_as_models_is_refused_for_multi_model_input(multi_model_cif: Path):
    """The models are already the ensemble."""
    with pytest.raises(ValueError, match="altlocs-as-models"):
        _load(multi_model_cif, altlocs_as_models=True)


def test_a_selection_matching_nothing_fails_loudly(ensemble_input):
    path, kwargs = ensemble_input
    with pytest.raises(ValueError):
        _load(path, row={"selection": "chain ZZ"}, **kwargs)
