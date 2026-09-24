"""
This file originated in RosettaCommons/foundry/models/rf3 and is licensed under BSD-3-Clause.
"""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.atom_array import (
    add_global_token_id_annotation,
    AddGlobalTokenIdAnnotation,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray, AtomArrayStack, stack
from jaxtyping import Bool, Float, Int
from loguru import logger

from sampleworks.metrics.metric import Metric
from sampleworks.utils.atom_array_utils import filter_to_common_atoms


# This method is copied from RosettaCommons/foundry/models/rf3/metrics/lddt.py, but
# modified to return residue-level LDDT scores as well.
def _calc_lddt(
    X_L: Float[torch.Tensor, "D L 3"],
    X_gt_L: Float[torch.Tensor, "D L 3"],
    crd_mask_L: Bool[torch.Tensor, "D L"],
    tok_idx: Int[torch.Tensor, "L"],  # noqa F821
    pairs_to_score: Bool[torch.Tensor, "L L"] | None = None,
    distance_cutoff: float = 15.0,
    eps: float = 1e-6,
    selected_token_ids: np.ndarray | None = None,
) -> tuple[Float[torch.Tensor, "D"], dict[str, list[float]]]:  # noqa F821
    """Calculates LDDT scores for each model in the batch.

    Parameters
    ----------
    X_L : Float[Tensor, "D L 3"]
        Predicted coordinates.
    X_gt_L : Float[Tensor, "D L 3"]
        Ground truth coordinates.
    crd_mask_L : Bool[Tensor, "D L"]
        Coordinate mask indicating valid atoms.
    tok_idx : Int[Tensor, "L"]
        Token index of each atom. Used to exclude same-token pairs.
    pairs_to_score : Bool[Tensor, "L L"] | None
        Boolean mask for pairs to score. If None, scores all valid pairs.
    distance_cutoff : float
        Distance cutoff for scoring pairs.
    eps : float
        Small epsilon to prevent division by zero.
    selected_token_ids : np.ndarray | None
        Optional subset of token IDs to compute residue-level scores for.

    Returns
    -------
    tuple[Float[Tensor, "D"], dict[str, list[float]]]
        LDDT scores for each model, and a dictionary of residue-level LDDT scores.
    """
    D, L = X_L.shape[:2]

    # Create pairs to score mask - if not provided, use upper triangular (includes diagonal)
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool).triu(0).to(X_L.device)
    else:
        assert pairs_to_score.shape == (L, L)
        pairs_to_score = pairs_to_score.triu(0).to(X_L.device)

    # Get indices of atom pairs to evaluate
    first_index: Int[torch.Tensor, "n_pairs"]  # noqa F821
    second_index: Int[torch.Tensor, "n_pairs"]  # noqa F821
    first_index, second_index = torch.nonzero(pairs_to_score, as_tuple=True)

    # Compute LDDT score for each model in the batch
    lddt_scores = []
    residue_level_lddt_scores = []
    # TODO: can this be further vectorized? https://github.com/k-chrispens/sampleworks/issues/50
    # i.e. there's not especially a reason for things to live inside a for loop that goes
    # through each batch / ensemble member individually
    for d in range(D):
        # Calculate pairwise distances in ground truth structure
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[d, first_index] - X_gt_L[d, second_index], dim=-1
        )

        # Create mask for valid pairs to score:
        # 1. Ground truth distance > 0 (atoms not at same position)
        # 2. Ground truth distance < cutoff (within interaction range)
        pair_mask = torch.logical_and(
            ground_truth_distances > 0, ground_truth_distances < distance_cutoff
        )

        # Only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[d, first_index] * crd_mask_L[d, second_index]

        # Don't score pairs that are in the same token (e.g., same residue)
        pair_mask *= tok_idx[first_index] != tok_idx[second_index]

        # Filter to only "valid" pairs
        valid_pairs = pair_mask.nonzero(as_tuple=True)

        pair_mask_valid = pair_mask[valid_pairs].to(X_L.dtype)
        ground_truth_distances_valid = ground_truth_distances[valid_pairs]

        first_index_valid: Int[torch.Tensor, "n_valid_pairs"] = first_index[  # noqa F821
            valid_pairs
        ]
        second_index_valid: Int[torch.Tensor, "n_valid_pairs"] = second_index[  # noqa F821
            valid_pairs
        ]

        # Calculate pairwise distances in predicted structure
        predicted_distances = torch.linalg.norm(
            X_L[d, first_index_valid] - X_L[d, second_index_valid], dim=-1
        )

        # Compute absolute distance differences (with small eps to avoid numerical issues)
        delta_distances = torch.abs(predicted_distances - ground_truth_distances_valid + eps)
        del predicted_distances, ground_truth_distances_valid

        # Compute the residue level metrics
        def get_lddt_distances_for_token(token_id):
            # I could be more clever with this, but I'd like to keep it easily readable.
            # the delta distances represent an upper triangular matrix, we need the symmetric form
            idx1u = tok_idx[first_index_valid] == token_id
            idx2u = tok_idx[second_index_valid] != token_id
            upper_triangle_result = delta_distances[idx1u & idx2u]

            idx1l = tok_idx[second_index_valid] == token_id
            idx2l = tok_idx[first_index_valid] != token_id
            lower_triangle_result = delta_distances[idx1l & idx2l]

            result = torch.hstack([upper_triangle_result, lower_triangle_result])

            lddt_score = (
                0.25
                * (
                    torch.sum(result < 0.5)
                    + torch.sum(result < 1.0)
                    + torch.sum(result < 2.0)
                    + torch.sum(result < 4.0)
                )
            # For now, I force the residue-level lDDT score to impute 0 when there are no paired
            # residues to score (i.e. result is an empty tensor). This is to match the global
            # lDDT behavior; but this prevents downstream operations from determining whether a score
            # is truly 0 (distances too far) or is NaN (there are no distances to assess). For example, there may be 
            # cases (e.g. bad selection string) where a token with no valid atoms is selected, in which
            # there are no distances to assess and NaN would have been returned. 
                / (len(result) + eps)
            )

            return lddt_score.item()

        # can this also be vectorized (over # of tokens) -> then reform into desired output format
        if selected_token_ids is not None:
            residue_lddt_dict = {
                tk.item(): get_lddt_distances_for_token(tk.item()) for tk in selected_token_ids
            }
        else:
            residue_lddt_dict = {
                tk.item(): get_lddt_distances_for_token(tk.item()) for tk in tok_idx.unique()
            }
        residue_level_lddt_scores.append(residue_lddt_dict)

        # TODO: is the *pair_mask_valid necessary? I think that mask is all 1 by construction.
        #   either that can change, or my get_lddt_distances_for_token is (possibly) wrong.
        # Calculate LDDT score using standard thresholds (0.5Å, 1.0Å, 2.0Å, 4.0Å)
        # LDDT is the average fraction of distances preserved within each threshold
        lddt_score = (
            0.25
            * (
                torch.sum((delta_distances < 0.5) * pair_mask_valid)  # 0.5Å threshold
                + torch.sum((delta_distances < 1.0) * pair_mask_valid)  # 1.0Å threshold
                + torch.sum((delta_distances < 2.0) * pair_mask_valid)  # 2.0Å threshold
                + torch.sum((delta_distances < 4.0) * pair_mask_valid)  # 4.0Å threshold
            )
            / (torch.sum(pair_mask_valid) + eps)  # Normalize by number of valid pairs
        )

        lddt_scores.append(lddt_score)

    # Map all the residue_level_lddt_scores from a list of dictionaries to a dictionary of lists
    residue_level_lddt_scores = {
        k: [d[k] for d in residue_level_lddt_scores] for k in residue_level_lddt_scores[0]
    }

    # return the token indices of the first axis of the distance diff matrix so that we
    # can use them to compute per-token DDT later.
    return torch.tensor(lddt_scores, device=X_L.device), residue_level_lddt_scores


def extract_lddt_features_from_atom_arrays(
    predicted_atom_array_stack: AtomArrayStack | AtomArray,
    ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
) -> dict[str, Any]:
    """Extract all features needed for LDDT computation from AtomArrays.

    Parameters
    ----------
    predicted_atom_array_stack : AtomArrayStack | AtomArray
        Predicted coordinates as AtomArray(Stack).
    ground_truth_atom_array_stack : AtomArrayStack | AtomArray
        Ground truth coordinates as AtomArray(Stack).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - ``X_L``: Predicted coordinates tensor (D, L, 3)
        - ``X_gt_L``: Ground truth coordinates tensor (D, L, 3)
        - ``crd_mask_L``: Coordinate validity mask (D, L)
        - ``tok_idx``: Token indices for each atom (L,)
        - ``chain_iid_token_lvl``: Chain identification at token level
    """
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)

    if (
        ground_truth_atom_array_stack.stack_depth() == 1
        and predicted_atom_array_stack.stack_depth() > 1
    ):
        # If the ground truth is a single model, and the predicted is a stack,
        # we need to expand the ground truth to the same length as the predicted
        ground_truth_atom_array_stack = stack(
            [ground_truth_atom_array_stack[0]] * predicted_atom_array_stack.stack_depth()
        )

    # Compute coordinates - convert AtomArrays to tensors
    X_L: Float[torch.Tensor, "D L 3"] = torch.from_numpy(predicted_atom_array_stack.coord).float()
    X_gt_L: Float[torch.Tensor, "D L 3"] = torch.from_numpy(
        ground_truth_atom_array_stack.coord
    ).float()

    # For the remaining feature generation, we can directly use the
    # first model in the stack (only coordinates are different)
    ground_truth_atom_array = ground_truth_atom_array_stack[0]

    # Create the coordinate mask using occupancy if available, fallback to coordinate validity
    # Note (marcus.collins@astera.org) added `is not None` check, hopefully this does not
    # change the behavior of the code
    if (
        "occupancy" in ground_truth_atom_array.get_annotation_categories()
        and ground_truth_atom_array.occupancy is not None
    ):
        # Use occupancy annotation (broadcast to all models in the stack) if present
        # (occupancy > 0 means atom is present)
        occupancy_mask = ground_truth_atom_array.occupancy > 0
        crd_mask_L: Bool[torch.Tensor, "D L"] = (
            torch.from_numpy(occupancy_mask).bool().unsqueeze(0).expand(X_gt_L.shape[0], -1)
        )
    else:
        # Fallback to coordinate validity (not NaN)
        crd_mask_L: Bool[torch.Tensor, "D L"] = ~torch.isnan(X_gt_L).any(dim=-1)

    # Get token indices using the same logic as ComputeAtomToTokenMap
    # Note (marcus.collins@astera.org) added `is not None` check. Hopefully this does not
    # change the behavior of the code
    if (
        "token_id" in ground_truth_atom_array.get_annotation_categories()
        and ground_truth_atom_array.token_id is not None
    ):
        # Use the existing token_id annotation (matches ComputeAtomToTokenMap exactly)
        tok_idx = ground_truth_atom_array.token_id.astype(np.int32)
    else:
        # Generate annotations with Transform pipeline
        pipe = Compose([AtomizeByCCDName(atomize_by_default=True), AddGlobalTokenIdAnnotation()])
        data = pipe({"atom_array": ground_truth_atom_array})
        tok_idx = data["atom_array"].token_id.astype(np.int32)

    # Compute chain identification at the token-level
    token_starts = get_token_starts(ground_truth_atom_array)

    if "chain_iid" in ground_truth_atom_array.get_annotation_categories():
        chain_iid_token_lvl = ground_truth_atom_array.chain_iid[token_starts]
    else:
        # Use the chain_id annotation instead
        # (e.g., for AF-3 outputs, where the chain_id is ostensibly the chain_iid)
        chain_iid_token_lvl = ground_truth_atom_array.chain_id[token_starts]

    return {
        "X_L": X_L,
        "X_gt_L": X_gt_L,
        "crd_mask_L": crd_mask_L,
        "tok_idx": tok_idx,
        "chain_iid_token_lvl": chain_iid_token_lvl,
    }


# Modified from original to optionally return residue-level lDDT for selected atoms
class AllAtomLDDT(Metric):
    """Computes all-atom LDDT scores from AtomArrays."""

    def __init__(self, log_lddt_for_every_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.log_lddt_for_every_batch = log_lddt_for_every_batch

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "selection": "selection",
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset({"selection"})

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        selection: str | None = None,
    ) -> dict[str, Any]:
        """Calculates all-atom LDDT between all pairs of atoms.

        Parameters
        ----------
        predicted_atom_array_stack : AtomArrayStack | AtomArray
            Predicted coordinates as AtomArray(Stack).
        ground_truth_atom_array_stack : AtomArrayStack | AtomArray
            Ground truth coordinates as AtomArray(Stack).
        selection : str | None
            Optional string to select which per-residue LDDT scores to report,
            using the AtomArray.mask() syntax.

        Returns
        -------
        dict[str, Any]
            A dictionary with all-atom LDDT scores:

            - ``best_of_1_lddt``: LDDT score for the first model
            - ``best_of_{N}_lddt``: Best LDDT score across all N models
            - ``residue_lddt_scores``: Per-residue LDDT scores
        """

        # Set token ids to something useful for residue-level LDDT (chain ID + residue number)
        # Note: add_global_token_id_annotation works with both AtomArray and
        # AtomArrayStack at runtime
        predicted_atom_array_stack = cast(
            AtomArrayStack,
            add_global_token_id_annotation(cast(Any, predicted_atom_array_stack)),
        )
        ground_truth_atom_array_stack = cast(
            AtomArrayStack,
            add_global_token_id_annotation(cast(Any, ground_truth_atom_array_stack)),
        )

        # restrict to atoms that are present in both structures
        _predicted_aa, _ground_truth_aa = filter_to_common_atoms(
            predicted_atom_array_stack, ground_truth_atom_array_stack
        )
        predicted_aa_stack = ensure_atom_array_stack(_predicted_aa)
        ground_truth_aa_stack = ensure_atom_array_stack(_ground_truth_aa)
        # FIXME: the next few lines are Claude-generated, and I think they're trivially true,
        #  I think this should test that ground_truth_aa_stack and predicted_aa_stack have the
        #  same number of atoms...
        if (
            predicted_aa_stack.array_length() != predicted_atom_array_stack.array_length()
            or ground_truth_aa_stack.array_length() != ground_truth_atom_array_stack.array_length()
        ):
            logger.warning("Chains did not exactly match between input AtomArrays.")

        if predicted_aa_stack.array_length() == 0:
            raise RuntimeError("No atoms in common between the two structures.")

        lddt_features = extract_lddt_features_from_atom_arrays(
            predicted_aa_stack, ground_truth_aa_stack
        )

        tok_idx = torch.tensor(lddt_features["tok_idx"]).to(lddt_features["X_L"].device)

        selected_token_ids = None
        if selection is not None:
            mask_fn = predicted_aa_stack.mask
            if mask_fn is None:
                raise RuntimeError(
                    "predicted_aa_stack does not support mask() You should read in atom arrays"
                    "using `atomworks.io.utils.io_utils.load_any()` to access this method"
                )
            mask = mask_fn(selection)
            selected_arr = cast(AtomArray, predicted_aa_stack[0, mask])
            selected_token_ids = selected_arr.token_id
            if selected_token_ids is not None:
                selected_token_ids = np.unique(selected_token_ids)

        all_atom_lddt, residue_level_lddt_scores = _calc_lddt(
            X_L=lddt_features["X_L"],
            X_gt_L=lddt_features["X_gt_L"],
            crd_mask_L=lddt_features["crd_mask_L"],
            tok_idx=tok_idx,
            pairs_to_score=None,  # By default, score all pairs, except those within the same token
            distance_cutoff=15.0,
            selected_token_ids=selected_token_ids,
        )

        chain_id = cast(np.ndarray, ground_truth_aa_stack.chain_id)
        res_id = cast(np.ndarray, ground_truth_aa_stack.res_id)
        token_id = cast(np.ndarray, ground_truth_aa_stack.token_id)
        unique_id = np.char.add(chain_id, res_id.astype(str))  # using '+' appears not to be robust
        token_to_residue_id_map = {k: v for k, v in zip(token_id, unique_id)}
        residue_level_lddt_scores = {
            str(token_to_residue_id_map[k]): v for k, v in residue_level_lddt_scores.items()
        }

        result = {
            "best_of_1_lddt": all_atom_lddt[0].item(),
            f"best_of_{len(all_atom_lddt)}_lddt": all_atom_lddt.max().item(),
            "residue_lddt_scores": residue_level_lddt_scores,
        }

        if self.log_lddt_for_every_batch:
            lddt_by_batch = {
                f"all_atom_lddt_{i}": all_atom_lddt[i].item() for i in range(len(all_atom_lddt))
            }
            result.update(lddt_by_batch)

        return result


class SelectedLDDT(Metric):
    """
    Calculates LDDT scores for a subset of atoms in the structure, defined by
    a selection string passed internally to AtomArrayStack.mask().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "selections": ("extra_info", "selections"),
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArray | AtomArrayStack,
        ground_truth_atom_array_stack: AtomArray | AtomArrayStack,
        selections: Iterable[str] = (),
    ) -> dict[str, dict[str, float | dict[str, list[float]]]]:
        """Calculates LDDT scores between a reference structure and predicted structure,
        calculated only using residues that match the given selection strings.

        Parameters
        ----------
        predicted_atom_array_stack : AtomArray | AtomArrayStack
            Predicted coordinates as AtomArray(Stack).
        ground_truth_atom_array_stack : AtomArray | AtomArrayStack
            Ground truth coordinates as AtomArray(Stack).
        selections : Iterable[str]
            Iterable of selection strings to pass to AtomArrayStack.mask.

        Returns
        -------
        dict[str, dict[str, float | dict[str, list[float]]]]
            Dictionary containing overall and residue-level LDDT scores for each selection.
        """

        # Compute interface LDDT scores for each selection
        results = {}
        # first filter each atom_array_stack:
        predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
        ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)

        for selection in selections:
            mask_fn_predicted = predicted_atom_array_stack.mask
            mask_fn_gt = ground_truth_atom_array_stack.mask
            if mask_fn_predicted is None or mask_fn_gt is None:
                raise RuntimeError("AtomArrayStack does not support mask()")
            mask_predicted = mask_fn_predicted(selection)
            filtered_predicted = cast(AtomArrayStack, predicted_atom_array_stack[:, mask_predicted])

            mask_ground_truth = mask_fn_gt(selection)
            filtered_ground_truth = cast(
                AtomArrayStack, ground_truth_atom_array_stack[:, mask_ground_truth]
            )

            # set the token ids, to avoid any possible confusion later on
            # Note: add_global_token_id_annotation works with AtomArrayStack at runtime
            filtered_predicted = cast(
                AtomArrayStack,
                add_global_token_id_annotation(cast(Any, filtered_predicted)),
            )
            filtered_ground_truth = cast(
                AtomArrayStack,
                add_global_token_id_annotation(cast(Any, filtered_ground_truth)),
            )

            lddt_features = extract_lddt_features_from_atom_arrays(
                filtered_predicted, filtered_ground_truth
            )
            # this is actually a per-atom "token". Not actually what it should be, I think.
            tok_idx = torch.tensor(lddt_features["tok_idx"]).to(lddt_features["X_L"].device)

            all_atom_lddt, residue_level_lddt_scores = _calc_lddt(
                X_L=lddt_features["X_L"],
                X_gt_L=lddt_features["X_gt_L"],
                crd_mask_L=lddt_features["crd_mask_L"],
                tok_idx=tok_idx,
                # By default, score all pairs, except those within the same token
                pairs_to_score=None,
                distance_cutoff=15.0,
            )

            # map the token ids back to the original selection, i.e. residue id and chain.
            chain_id = cast(np.ndarray, filtered_predicted.chain_id)
            res_id = cast(np.ndarray, filtered_predicted.res_id)
            token_id = cast(np.ndarray, filtered_predicted.token_id)
            unique_id = np.char.add(chain_id, res_id.astype(str))
            token_to_residue_id_map = {k: v for k, v in zip(token_id, unique_id)}
            residue_level_lddt_scores = {
                str(token_to_residue_id_map[k]): v for k, v in residue_level_lddt_scores.items()
            }

            results[selection] = {
                "overall_lddt": all_atom_lddt.detach().cpu().numpy(),
                "residue_lddt_scores": residue_level_lddt_scores,
            }

        return results
