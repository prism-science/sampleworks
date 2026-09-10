"""Selection and stripping on multi-model input.

A stack shares its per-atom annotations across models, so one boolean mask
describes every model and the repo's existing helpers take a stack directly.
The generator used to refuse these options on multi-model files; these pin that
they apply, that every model survives, and that the two options a stack really
cannot express are still refused.
"""

from __future__ import annotations

import numpy as np
import pytest
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx import CIFFile, set_structure


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

from pathlib import Path

from sampleworks.synthetic.generate_synthetic_sf import BatchRowForMTZ
from sampleworks.synthetic.generate_synthetic_sf_lunus import load_configurations


@pytest.fixture(scope="module")
def multi_model_cif(resources_dir: Path, tmp_path_factory) -> Path:
    """A genuine two-model CIF, written from 1VME's two altloc conformations.

    The repo ships no multi-model structure, and the point of these tests is the
    real loader path, so the file is built rather than mocked.
    """
    source = resources_dir / "1vme" / "1vme_final.cif"
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")

    atom_array, coords = load_configurations(
        source, BatchRowForMTZ(filename=source.name), "default", altlocs_as_models=True
    )
    assert coords.shape[0] >= 2, "1VME should expand to at least two conformations"

    stack = AtomArrayStack(coords.shape[0], coords.shape[1])
    for field in ("chain_id", "res_id", "res_name", "atom_name", "element", "hetero"):
        stack.set_annotation(field, getattr(atom_array, field))
    stack.coord = coords.astype(np.float32)

    path = tmp_path_factory.mktemp("multi_model") / "ensemble.cif"
    cif = CIFFile()
    set_structure(cif, stack)
    cif.write(path)
    return path


def _load(path: Path, **kwargs):
    return load_configurations(
        path, BatchRowForMTZ(filename=path.name, **kwargs.pop("row", {})), "default", **kwargs
    )


def test_multi_model_input_loads_every_model(multi_model_cif: Path):
    _, coords = _load(multi_model_cif)

    assert coords.shape[0] >= 2, "all models must survive the load"


def test_selection_applies_to_multi_model_input(multi_model_cif: Path):
    """The claim this replaces was that selection needed 'index plumbing that
    does not exist'. It does not: the mask is shared across models."""
    _, full = _load(multi_model_cif)
    atom_array, selected = _load(multi_model_cif, row={"selection": "chain A"})

    assert selected.shape[0] == full.shape[0], "selection must not drop models"
    assert selected.shape[1] < full.shape[1], "selection must drop atoms"
    assert set(np.asarray(atom_array.chain_id)) == {"A"}


def test_stripping_hydrogens_applies_to_multi_model_input(multi_model_cif: Path):
    atom_array, coords = _load(multi_model_cif, strip_hydrogens=True)

    assert "H" not in set(np.asarray(atom_array.element))
    assert coords.shape[0] >= 2


def test_stripping_waters_applies_to_multi_model_input(multi_model_cif: Path):
    _, full = _load(multi_model_cif)
    atom_array, coords = _load(multi_model_cif, strip_waters=True)

    assert "HOH" not in set(np.asarray(atom_array.res_name))
    assert coords.shape[0] == full.shape[0]


def test_topology_and_coordinates_stay_in_register_after_filtering(multi_model_cif: Path):
    """The returned topology must describe the returned coordinates, per model."""
    atom_array, coords = _load(multi_model_cif, row={"selection": "chain A"})

    assert len(atom_array) == coords.shape[1]


def test_occupancy_mode_is_still_refused_for_multi_model_input(multi_model_cif: Path):
    """A stack has one shared occupancy annotation, so a per-model answer has
    nowhere to go. Marcus conceded this one."""
    with pytest.raises(ValueError, match="occupancy-mode"):
        load_configurations(
            multi_model_cif,
            BatchRowForMTZ(filename=multi_model_cif.name),
            "custom",
        )


def test_altlocs_as_models_is_still_refused_for_multi_model_input(multi_model_cif: Path):
    """The models already are the ensemble; expanding altlocs on top is not a
    meaningful composition."""
    with pytest.raises(ValueError, match="altlocs-as-models"):
        _load(multi_model_cif, altlocs_as_models=True)


def test_a_selection_matching_nothing_fails_loudly(multi_model_cif: Path):
    with pytest.raises(ValueError):
        _load(multi_model_cif, row={"selection": "chain ZZ"})
