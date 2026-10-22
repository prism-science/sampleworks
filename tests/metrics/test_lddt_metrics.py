"""
Tests for lDDT metrics
"""

from typing import cast

import numpy as np
import pytest
from sampleworks.metrics.lddt import AllAtomLDDT, SelectedLDDT


# These tests are currently too high level, but they will serve for now to demonstrate
# the expected behavior and make sure nothing gets broken.


@pytest.mark.gpu
def test_all_atom_lddt_end_to_end(altlocA_backbone, altlocB_backbone):
    selection_string = "res_id > 179 and res_id < 190"
    allatom = AllAtomLDDT()
    results = allatom.compute(altlocA_backbone, altlocB_backbone, selection_string)

    expected_results = {
        "best_of_1_lddt": 0.970,
        "residue_lddt_scores": {
            "A180": [0.6408],
            "A181": [0.5700],
            "A182": [0.4511],
            "A183": [0.6169],
            "A184": [0.7267],
            "A185": [0.6974],
            "A186": [0.5572],
            "A187": [0.7665],
            "A188": [0.8666],
            "A189": [0.8819],
        },
    }

    assert "best_of_1_lddt" in results
    assert "residue_lddt_scores" in results

    # Check best_of_1_lddt value
    assert results["best_of_1_lddt"] == pytest.approx(expected_results["best_of_1_lddt"], abs=0.002)

    # Check that all expected keys are present in residue_lddt_scores
    assert set(results["residue_lddt_scores"].keys()) == set(
        expected_results["residue_lddt_scores"].keys()
    )

    # Check each residue's LDDT scores
    for residue_key in expected_results["residue_lddt_scores"]:
        assert residue_key in results["residue_lddt_scores"], f"Missing residue: {residue_key}"

        result_scores = results["residue_lddt_scores"][residue_key]
        expected_scores = expected_results["residue_lddt_scores"][residue_key]

        # Check that the list lengths match
        assert len(result_scores) == len(expected_scores), (
            f"Length mismatch for {residue_key}: got {len(result_scores)}, expected "
            f"{len(expected_scores)}"
        )

        # Check each score value
        for i, (result_score, expected_score) in enumerate(zip(result_scores, expected_scores)):
            assert result_score == pytest.approx(expected_score, abs=0.005), (
                f"Score mismatch for {residue_key}[{i}]: got {result_score}, expected "
                f"{expected_score}\nAll resulting scores: {results['residue_lddt_scores']}\n"
                f"Expected scores: {expected_results['residue_lddt_scores']}"
            )


@pytest.mark.gpu
def test_selected_lddt_end_to_end(altlocA_backbone, altlocB_backbone):
    import numpy as np

    selection_string = "res_id > 179 and res_id < 185"
    lddt = SelectedLDDT()
    results = lddt.compute(altlocA_backbone, altlocB_backbone, (selection_string,))

    expected_results = {
        selection_string: {
            "overall_lddt": np.array([0.8281]),
            "residue_lddt_scores": {
                "A180": [0.796],
                "A181": [0.8594],
                "A182": [0.7969],
                "A183": [0.8359],
                "A184": [0.8516],
            },
        }
    }

    assert selection_string in results
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    result_dict = results[selection_string]
    expected_dict = expected_results[selection_string]
    # Check overall value - now returns numpy array
    assert isinstance(result_dict["overall_lddt"], np.ndarray), (
        "overall_lddt should be a numpy array"
    )
    overall_lddt = np.asarray(result_dict["overall_lddt"], dtype=float)
    expected_overall_lddt = np.asarray(expected_dict["overall_lddt"], dtype=float)
    np.testing.assert_allclose(overall_lddt, expected_overall_lddt, atol=0.001)

    # Check that all expected keys are present in residue_lddt_scores
    result_residue_scores = cast(dict[str, list[float]], result_dict["residue_lddt_scores"])
    expected_residue_scores = cast(dict[str, list[float]], expected_dict["residue_lddt_scores"])
    assert set(result_residue_scores.keys()) == set(expected_residue_scores.keys())

    # Check each residue's LDDT scores
    for residue_key in expected_residue_scores:
        result_scores = result_residue_scores[residue_key]
        expected_scores = expected_residue_scores[residue_key]

        # Check that the list lengths match
        assert len(result_scores) == len(expected_scores), (
            f"Length mismatch for {residue_key}: got {len(result_scores)}, expected "
            f"{len(expected_scores)}"
        )

        # Check each score value
        for i, (result_score, expected_score) in enumerate(zip(result_scores, expected_scores)):
            assert result_score == pytest.approx(expected_score, abs=0.001), (
                f"Score mismatch for {residue_key}[{i}]: got {result_score}, expected "
                f"{expected_score}"
            )


# Unit tests for sampleworks.metrics.lddt._calc_lddt

import torch
from sampleworks.metrics.lddt import _calc_lddt


# These tests originated in RosettaCommons/foundry/models/rf3 and is licensed under BSD-3-Clause.
# For each case, added test for expected token-level outputs (where in this case, token = atom)


def _coords(points: list[list[float]]) -> torch.Tensor:
    """One model with the given atom coordinates → shape (1, L, 3)."""
    return torch.tensor([points], dtype=torch.float32)


def test_perfect_prediction_scores_one():
    """All distance differences are 0 -> lDDT of 1 expected"""
    coords = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    tok = torch.arange(4)
    lddt, lddt_residues = _calc_lddt(coords, coords.clone(), mask, tok)

    # single
    assert lddt.shape == (1,)
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)

    # token-level
    assert len(lddt_residues) == 4
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]  # take first value, since batch dim size = 1
        assert token_score == pytest.approx(1, abs=1e-4), (
            f"Score mismatch for {token_name}: got {token_score}, expected 1"
        )


def test_large_error_scores_zero():
    """All distance differences are greater than 4Å (largest threshold) -> lDDT of 0 expected""" ""
    gt = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    pred = gt * 10.0  # every pairwise distance off by far more than 4 Å
    mask = torch.ones(1, 4, dtype=torch.bool)
    tok = torch.arange(4)
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, tok)

    # single
    assert torch.allclose(lddt, torch.zeros(1), atol=1e-4)

    # token-level
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]
        assert token_score == pytest.approx(0, abs=1e-4), (
            f"Score mismatch for {token_name}: got {token_score}, expected 0"
        )


@pytest.mark.parametrize("masked_atom", [0, 3], ids=["first_atom", "last_atom"])
def test_unresolved_atoms_are_masked_out(masked_atom: int):
    """An unresolved atom's pairs are dropped wherever the atom sits in a pair.

    Pairs are scored once as (i, j) with i < j, so atom 0 is only ever the first atom of
    a pair and atom 3 only ever the second. The two cases cover both halves of the mask.

    Parameters
    ----------
    masked_atom : int
        Index of the unresolved atom, which is also badly placed in the prediction.
    """
    gt = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    pred = gt.clone()
    pred[0, masked_atom] = torch.tensor([100.0, 0, 0])  # badly placed...
    # ...but unresolved, so all of its pairs are dropped from scoring.
    mask = torch.ones(1, 4, dtype=torch.bool)
    mask[0, masked_atom] = False
    tok = torch.arange(4)
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, tok)

    # single
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)

    # token-level
    # This simulates the case where selected_token_ids tries to select a token (residue) for which
    # all atoms are invalid, which may(?) happen since crd_mask_L and selected_token_ids
    # do not check against each other. In this case, we output NaN to differentiate from a
    # true measured lDDT of 0.
    assert lddt_residues.keys() == {0, 1, 2, 3}
    for token_name, token_scores in lddt_residues.items():
        token_score = token_scores[0]

        # the unresolved atom has no scored pairs
        if token_name == masked_atom:
            assert np.isnan(token_score), (
                f"Score mismatch for {token_name}: got {token_score}, expected NaN"
            )
        else:
            # valid residues should score 1
            assert token_score == pytest.approx(1, abs=1e-4), (
                f"Score mismatch for {token_name}: got {token_score}, expected 1"
            )


def test_same_token_pairs_excluded():
    """Test whether same token pairs are correctly excluded from pair distance calculation"""
    gt = _coords([[0, 0, 0], [1, 0, 0]])
    pred = gt.clone()  # perfect
    mask = torch.ones(1, 2, dtype=torch.bool)

    # Distinct tokens: the single 1 Å pair is scored → 1.0.
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, torch.tensor([0, 1]))
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)
    assert len(lddt_residues) == 2
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]
        assert token_score == pytest.approx(1, abs=1e-4), (
            f"Score mismatch for {token_name}: got {token_score}, expected 1"
        )

    # Same token: the only pair is excluded → no valid pairs → 0.0.
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, torch.tensor([0, 0]))
    assert np.isnan(lddt)
    assert len(lddt_residues) == 1
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]
        assert np.isnan(token_score), (
            f"Score mismatch for {token_name}: got {token_score}, expected NaN"
        )


def test_distance_cutoff_excludes_far_pairs():
    """Test whether distance_cutoff restraint correctly removes pairs too far apart"""
    gt = _coords([[0, 0, 0], [20, 0, 0]])  # 20 Å apart
    pred = gt.clone()
    mask = torch.ones(1, 2, dtype=torch.bool)
    tok = torch.tensor([0, 1])

    # Default cutoff 15 Å → pair is out of range → excluded → 0.0.
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, tok)
    assert np.isnan(lddt)
    assert len(lddt_residues) == 2
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]
        assert np.isnan(token_score), (
            f"Score mismatch for {token_name}: got {token_score}, expected NaN"
        )

    # Cutoff 30 Å → pair is in range and perfect → 1.0.
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, tok, distance_cutoff=30.0)
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)
    assert len(lddt_residues) == 2
    for token_name in lddt_residues:
        token_score = lddt_residues[token_name][0]
        assert token_score == pytest.approx(1, abs=1e-4), (
            f"Score mismatch for {token_name}: got {token_score}, expected 1"
        )


def test_batched_models():
    """Each model is scored against its own "ground truth" with its mask.

    Model 0 is 3 atoms with 1 Å spacing with every atom resolved: 3 pairs,
    all passing. Model 1's ground truth has 2 Å spacing and atom 0 unresolved, leaving
    only pair (1, 2), predicted 0.75 Å too long: it passes the 1/2/4 Å thresholds, so
    gets lddt score of 0.75. Model 1 has fewer scored pairs than model 0, so its pair list is padded
    """
    gt = torch.tensor(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [2, 0, 0], [4, 0, 0]],
        ],
        dtype=torch.float32,
    )  # (2, 3, 3)
    pred = gt.clone()
    pred[1, 0] = torch.tensor([50.0, 0, 0])  # unresolved, so its misplacement is ignored
    pred[1, 2] = torch.tensor([4.75, 0, 0])  # pair (1, 2) is 2.75 Å, 0.75 Å off
    mask = torch.tensor([[True, True, True], [False, True, True]])
    tok = torch.arange(3)
    lddt, lddt_residues = _calc_lddt(pred, gt, mask, tok)

    # single
    torch.testing.assert_close(lddt, torch.tensor([1.0, 0.75]))

    # Per-token scores, one entry per model. Atom 0 is unresolved in model 1, so it
    # appears in no scored pair there and its score is NaN.
    expected_residues = {0: [1.0, np.nan], 1: [1.0, 0.75], 2: [1.0, 0.75]}
    assert lddt_residues.keys() == expected_residues.keys()
    for token_name, expected_scores in expected_residues.items():
        np.testing.assert_allclose(
            lddt_residues[token_name], expected_scores, atol=1e-6, err_msg=f"token {token_name}"
        )
