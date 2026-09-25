"""Regression tests for RF3 atom ordering consistency.

Verifies that ``model_atom_array`` built during ``RF3Wrapper.featurize()`` has
the same atom count and ordering as the model's internal feature tensors
(``atom_to_token_map``).
"""

import numpy as np
import pytest
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE


if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import annotate_structure_for_rf3


@pytest.mark.gpu
@pytest.mark.slow
class TestRF3AtomOrdering:
    """Validate model_atom_array matches model feature dimensions."""

    @require_rf3()
    @pytest.mark.parametrize(
        "structure_fixture",
        [
            "structure_5i09_density",
            "structure_5sop_density",
            "structure_9bn8_density",
            "structure_6ni6_density",
        ],
    )
    def test_model_atom_array_matches_feature_count(self, rf3_wrapper, structure_fixture, request):
        """model_atom_array atom count must match atom_to_token_map length exactly.

        Regression test for 5I09 where OXT atoms at chain breaks were retained
        in model_atom_array but absent from the model's
        internal atom accounting, causing coordinate misalignment.
        """
        structure = request.getfixturevalue(structure_fixture)
        annotated = annotate_structure_for_rf3(structure)
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning

        assert cond.model_atom_array is not None

        num_feature_atoms = len(cond.features["atom_to_token_map"])
        assert len(cond.model_atom_array) == num_feature_atoms, (
            f"model_atom_array has {len(cond.model_atom_array)} atoms but "
            f"atom_to_token_map has {num_feature_atoms}. Coordinate mapping "
            "will be misaligned."
        )

    @require_rf3()
    @pytest.mark.parametrize(
        "structure_fixture",
        [
            "structure_5i09_density",
            "structure_5sop_density",
            "structure_9bn8_density",
            "structure_6ni6_density",
        ],
    )
    def test_no_oxt_atoms_in_model_atom_array(self, rf3_wrapper, structure_fixture, request):
        """model_atom_array must not contain OXT atoms.

        The pipeline's ``RemoveTerminalOxygen`` transform removes these atoms.
        """
        structure = request.getfixturevalue(structure_fixture)
        annotated = annotate_structure_for_rf3(structure)
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning

        assert cond.model_atom_array is not None
        oxt_count = int(np.sum(cond.model_atom_array.atom_name == "OXT"))
        assert oxt_count == 0, (
            f"model_atom_array contains {oxt_count} OXT atoms that the pipeline "
            "should have removed."
        )

    @require_rf3()
    @pytest.mark.parametrize(
        "structure_fixture",
        [
            "structure_5i09_density",
            "structure_5sop_density",
            "structure_9bn8_density",
            "structure_6ni6_density",
        ],
    )
    def test_no_hydrogen_atoms_in_model_atom_array(self, rf3_wrapper, structure_fixture, request):
        """model_atom_array must not contain hydrogen atoms.

        The pipeline's ``RemoveHydrogens`` transform removes these.
        """
        structure = request.getfixturevalue(structure_fixture)
        annotated = annotate_structure_for_rf3(structure)
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning

        assert cond.model_atom_array is not None
        h_count = int(np.sum(cond.model_atom_array.element == "H"))
        assert h_count == 0, (
            f"model_atom_array contains {h_count} hydrogen atoms that the pipeline "
            "should have removed."
        )
