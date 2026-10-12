"""Tests for build_reference_frames in both reference layouts."""

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, from_template
from sampleworks.eval.grid_search_eval_utils import build_reference_frames

from tests.utils.atom_array_builders import build_test_atom_array


SELECTION = "chain_id == 'A' and res_id == 2"


def _build_reference_atoms(altloc_ids: list[str], res_ids: list[int], coords: np.ndarray):
    """Build a one-CA-per-entry AtomArray carrying the given altloc IDs.

    Parameters
    ----------
    altloc_ids
        Per-atom altloc ID.
    res_ids
        Per-atom residue ID, all on chain A.
    coords
        Atom coordinates, shape ``(n_atoms, 3)``.

    Returns
    -------
    AtomArray
        Atom array with ``altloc_id``, ``occupancy`` and ``b_factor`` annotations.
    """
    atom_array = build_test_atom_array(
        chain_ids=["A"] * len(res_ids),
        res_ids=res_ids,
        atom_names=["CA"] * len(res_ids),
        coords=coords,
    )
    atom_array.set_annotation("altloc_id", np.array(altloc_ids))
    return atom_array


@pytest.fixture
def single_model_reference() -> AtomArray:
    """Deposited-style reference: anchor res 1, altlocs A/B on res 2 (selected) and res 3."""
    coords = np.arange(15, dtype=np.float32).reshape(5, 3)
    return _build_reference_atoms(["", "A", "B", "A", "B"], [1, 2, 2, 3, 3], coords)


@pytest.fixture
def multi_model_reference() -> AtomArrayStack:
    """Ensemble reference: two models, anchor res 1, altloc X on res 2 (selected) and res 3."""
    template = _build_reference_atoms(["", "X", "X"], [1, 2, 3], np.zeros((3, 3), dtype=np.float32))
    # (n_models, n_atoms, 3): every coordinate differs between models, anchors included.
    coords = np.stack(
        [
            np.arange(9, dtype=np.float32).reshape(3, 3),
            100 + np.arange(9, dtype=np.float32).reshape(3, 3),
        ]
    )
    return from_template(template, coords)


class TestSingleModelReference:
    def test_one_frame_per_altloc(self, single_model_reference):
        frames = build_reference_frames(single_model_reference, SELECTION)
        assert frames.stack_depth() == 2

    def test_keeps_anchors_and_selection_only(self, single_model_reference):
        frames = build_reference_frames(single_model_reference, SELECTION)
        np.testing.assert_array_equal(frames.res_id, [1, 2])

    def test_each_frame_holds_its_altloc_coordinates(self, single_model_reference):
        frames = build_reference_frames(single_model_reference, SELECTION)
        coords = single_model_reference.coord
        np.testing.assert_array_equal(frames.coord[0], coords[[0, 1]])  # anchor + altloc A
        np.testing.assert_array_equal(frames.coord[1], coords[[0, 2]])  # anchor + altloc B

    def test_selection_matching_nothing_raises(self, single_model_reference):
        with pytest.raises(ValueError, match="res_id == 99"):
            build_reference_frames(single_model_reference, "res_id == 99")


class TestMultiModelReference:
    def test_one_frame_per_model(self, multi_model_reference):
        frames = build_reference_frames(multi_model_reference, SELECTION)
        assert frames.stack_depth() == 2

    def test_keeps_anchors_and_selection_only(self, multi_model_reference):
        frames = build_reference_frames(multi_model_reference, SELECTION)
        np.testing.assert_array_equal(frames.res_id, [1, 2])
        np.testing.assert_array_equal(frames.altloc_id, ["", "X"])

    def test_preserves_model_order_and_coordinates(self, multi_model_reference):
        frames = build_reference_frames(multi_model_reference, SELECTION)
        # Each frame keeps its own model's anchor and selection coordinates, in model order.
        np.testing.assert_array_equal(frames.coord, multi_model_reference.coord[:, [0, 1]])

    def test_selection_matching_nothing_raises(self, multi_model_reference):
        """Anchors alone are non-empty, so the check must be on the selection itself."""
        with pytest.raises(ValueError, match="No atoms in reference match selection"):
            build_reference_frames(multi_model_reference, "res_id == 99")
