"""Regression tests for RF3 atom ordering consistency.

Verifies that ``model_atom_array`` built during ``RF3Wrapper.featurize()`` has
the same atom count and ordering as the model's internal feature tensors
(``atom_to_token_map``). Additionally verifies that unresolved atoms are placed
on the closest resolved atom/token in the ``coord_to_be_noised`` annotation.
"""

import numpy as np
import pytest
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE
from biotite.structure import rmsd, filter_intersection
from atomworks.io.parser import parse_atom_array


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

@pytest.mark.gpu
@pytest.mark.slow
class TestRF3UnresolvedAtom:
    '''Validate that RF3 correctly handles placing unresolved atoms to the nearest
    atom/token in the coord_to_be_noised annotation
    '''

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
    def test_coord_to_be_noised_in_model_atom_array(self, rf3_wrapper, structure_fixture, request):
        """model_atom_array has valid .coord_to_be_noised annotations.

        RF3.wrapper.featurize() should place cleaned coordinates under the .coord_to_be_noised annotation
        rather than modifying the ground truth .coord attribute.
        """
        # Pass structure through RF3 Wrapper featurize()
        structure = request.getfixturevalue(structure_fixture)
        annotated = annotate_structure_for_rf3(structure)
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning
        assert cond.model_atom_array is not None

        # Check .coord_to_be_noised annotation exist
        assert "coord_to_be_noised" in cond.model_atom_array.get_annotation_categories()

        # Check .coord_to_be_noised no longer has any unresolved atoms
        cleaned_coords = cond.model_atom_array.coord_to_be_noised
        unresolved_coord_to_be_noised_mask = (
            (np.isnan(cleaned_coords))  # >= 1 NaN coordinate
            | (cleaned_coords <= 0.0)  # 0 occupancy
        )
        assert np.sum(unresolved_coord_to_be_noised_mask) <= 0, (
            "model.coord_to_be_noised has unplaced unresolved atoms"
        )

    @require_rf3()
    def test_unresolved_atom_placement(
        self, 
        rf3_wrapper, 
        structure_5i09_prediction,
        structure_5i09_prediction_deleted_atoms,
        structure_5i09_prediction_deleted_atoms_idxs
    ):
        """RF3.wrapper.featurize() should place all unresolved atoms on a nearby resolved atom/token. This
        behavior is indirectly tested for by checking whether the RMSD(nearest atom/token placement, true structure)
        < RMSD (centroid placement, true structure).

        This test utilizes structure_5i09_prediction_deleted_atoms, which is structure_5i09_prediction (one
        model only) with 6 atoms deleted from the structure. Identifying annotations for the deleted atoms are
        listed in structure_5i09_prediction_deleted_atoms_idxs.
        """
        # Rename fixtures for convenience
        structure = structure_5i09_prediction
        structure_unresolved = structure_5i09_prediction_deleted_atoms
        unresolved_atom_idxs = structure_5i09_prediction_deleted_atoms_idxs

        # (1) Pass structure through RF3 Wrapper featurize()
        # We expect this to place unresolved atoms to nearest atom/token.
        annotated_unresolved = annotate_structure_for_rf3(structure_unresolved)
        features_unresolved = rf3_wrapper.featurize(annotated_unresolved)
        test_unresolved_model = features_unresolved.conditioning.model_atom_array

        for unresolved_atom_idx in unresolved_atom_idxs:

            # Get target atom
            unresolved_atom_array = test_unresolved_model[
                (test_unresolved_model.chain_id == unresolved_atom_idx['chain_id'])
                & (test_unresolved_model.res_id == unresolved_atom_idx['res_id'])
                & (test_unresolved_model.atom_name == unresolved_atom_idx['atom_name'])
            ]
            assert len(unresolved_atom_array) == 1, "More than 1 atom matched description"
            unresolved_atom = unresolved_atom_array[0]

            # Unresolved atoms are resolved in coord_to_be_noised annotation
            assert not np.any(np.isnan(
                unresolved_atom.coord_to_be_noised
            )),(
                f'{unresolved_atom_idx} is not resolved in coord_to_be_noised'
            )
            assert unresolved_atom.occupancy > 0, (
                f'{unresolved_atom_idx} is not resolved in coord_to_be_noised'
            )

            # and unchanged in the coord annotation
            assert np.any(
                np.isnan(unresolved_atom.coord) 
                | (unresolved_atom.coord == -1) # RF3 inference pipeline imputes NaNs as -1
            ), (
                f'{unresolved_atom_idx} is resolved in coord'
            )

        # (2) Alternatively, we test against previous centroid placement strategy
        centroid_unresolved_model = structure_unresolved['asym_unit'][0]

        # add missing atoms manually, since we do not run through inference pipeline
        centroid_unresolved_model = parse_atom_array(
            centroid_unresolved_model,
            add_missing_atoms=True,
            hydrogen_policy='keep'
        )['asym_unit'][0]

        # calculate centroid
        centroid = np.nanmean(centroid_unresolved_model.coord, axis=0)
        centroid_unresolved_mask = (~np.isfinite(centroid_unresolved_model.coord))

        # replace any unresolved with centroid
        centroid_unresolved_model.coord[centroid_unresolved_mask] = np.broadcast_to(
            centroid,
            centroid_unresolved_model.coord.shape
        )[centroid_unresolved_mask]

        # then set occupancy to 1 and b-factor to 20 (to match atoms from reference structure)
        # (this would need to change if default b-factor level changes)
        centroid_unresolved_model.occupancy[:] = 1
        centroid_unresolved_model.b_factor[centroid_unresolved_mask.any(axis=-1)] = 20

        # (3) Find RMSD of closest atom/token strategy
        # only measure RMSD between atoms in both structures
        reference_test_mask = filter_intersection(
            array = structure['asym_unit'][0],
            intersect = test_unresolved_model
        )
        # Double check that the mask includes the 6 atoms which were manually deleted ("unresolved")
        # from the original structure
        assert reference_test_mask.sum() - (
            filter_intersection(
                structure_unresolved['asym_unit'][0],
                test_unresolved_model
            ).sum()) == 6, (
                "Number of manually removed atoms does not match constructed expectation"
            )
        # (create two masks of atom intersection here, since AtomArrays are different lengths)
        test_reference_mask = filter_intersection(
            array = test_unresolved_model,
            intersect = structure['asym_unit'][0]
        )
        # calculate RMSD
        test_rmsd = rmsd(
            reference = structure['asym_unit'][0].coord[reference_test_mask],
            subject = test_unresolved_model.coord_to_be_noised[test_reference_mask]
        )

        # (4) RMSD of centroid strategy
        reference_centroid_mask = filter_intersection(
            structure['asym_unit'][0],
            centroid_unresolved_model
        )
        assert reference_centroid_mask.sum() - (
            filter_intersection(
                structure_unresolved['asym_unit'][0],
                centroid_unresolved_model
            ).sum()) == 6, (
                "Number of manually removed atoms does not match constructed expectation"
            )
        centroid_reference_mask = filter_intersection(
            centroid_unresolved_model,
            structure['asym_unit'][0]
        )
        centroid_rmsd = rmsd(
            reference = structure['asym_unit'][0].coord[reference_centroid_mask],
            subject = centroid_unresolved_model[centroid_reference_mask]
        )

        # (5) RMSD of closest atom/token strategy should be lower (when compared to original structure with
        # all atoms present), since we are not moving unresolved atom as far from its ground truth position
        assert test_rmsd < centroid_rmsd, (
            "Placing unresolved atoms on centroid has higher RMSD than nearest token/atom",
            f'RMSD: nearest atom/token {test_rmsd:.3f} vs centroid {centroid_rmsd:.3f}'
        )