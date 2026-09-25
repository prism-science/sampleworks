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
from sampleworks.metrics.metric_utils import pad_ragged
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
) -> tuple[Float[torch.Tensor, "D"], dict[int, list[float]]]:  # noqa F821
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
    tuple[Float[Tensor, "D"], dict[int, list[float]]]
        LDDT scores for each model, and a dictionary of residue-level LDDT scores.
    """
    # Construct pairs_to_score as all possible pairs if not provided
    D, L = X_L.shape[:2]
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool).triu(0).to(X_L.device)
    else:
        assert pairs_to_score.shape == (L, L)
        pairs_to_score = pairs_to_score.triu(0).to(X_L.device)

    # Get indices of atom pairs to evaluate
    first_index: Int[torch.Tensor, "n_pairs"]  # noqa F821
    second_index: Int[torch.Tensor, "n_pairs"]  # noqa F821
    first_index, second_index = torch.nonzero(pairs_to_score, as_tuple=True)

    # Calculate delta distances for valid pairs from each model in batch
    delta_distances, first_index_valid, second_index_valid, valid_mask = find_valid_distances(
        X_L, X_gt_L, first_index, second_index, crd_mask_L, tok_idx, distance_cutoff
    )  # all are (D, N)

    # Pre-calculate threshold passes for each pair
    # (To be used by both global and residue-level lDDT calculation)
    threshold_passes = torch.stack(
        [
            delta_distances < 0.5,  # 0.5Å threshold
            delta_distances < 1.0,  # 1.0Å threshold
            delta_distances < 2.0,  # 2.0Å threshold
            delta_distances < 4.0,  # 4.0Å threshold
        ],
        dim=-1,
    )  # (D, N, 4)

    # Remove contributions of padding elements
    threshold_passes &= valid_mask[..., None]  # (D, N, 4)
    threshold_passes = threshold_passes.sum(dim=-1)  # Flatten alongst threshold dimension

    # Calculate global lDDT score
    lddt_score = (
        0.25
        * threshold_passes.sum(dim=-1)  # For global, sum over pairs
        # Adding eps to denominator here makes having no valid pairs return 0 instead of NaN
        / (valid_mask.sum(dim=-1) + eps)  # Normalize by number of valid pairs
    )

    # Construct selected_token_ids if not provided
    if selected_token_ids is None:
        selected_token_ids_tensor = tok_idx.unique()
    else:
        # convert type to match above case
        selected_token_ids_tensor = torch.as_tensor(
            selected_token_ids, dtype=tok_idx.dtype, device=X_L.device
        )

    # Calculate residue level lDDT scores
    residue_level_lddt_scores = torch.vmap(
        residue_level_lddt_single, in_dims=(0, 0, 0, 0, None, None, None)
    )(
        threshold_passes,  # vectorize over batch
        first_index_valid,  # vectorize over batch
        second_index_valid,  # vectorize over batch
        valid_mask,  # vectorize over batch
        tok_idx,  # pass directly
        selected_token_ids_tensor,  # pass directly, needs to be tensor for vmap
        eps,
    )  # (D, n_tokens)

    # Reformat into dictionary of lists
    residue_level_lddt_scores = {
        int(token_id.item()): cast(list[float], residue_level_lddt_scores[:, i].tolist())
        for i, token_id in enumerate(selected_token_ids_tensor)
    }

    # return the token indices of the first axis of the distance diff matrix so that we
    # can use them to compute per-token DDT later.
    return lddt_score, residue_level_lddt_scores


def find_valid_distances(
    X_L: Float[torch.Tensor, "D L 3"],
    X_gt_L: Float[torch.Tensor, "D L 3"],
    first_index: Int[torch.Tensor, "n_pairs"],  # noqa F821
    second_index: Int[torch.Tensor, "n_pairs"],  # noqa F821
    crd_mask_L: Bool[torch.Tensor, "D L"],
    tok_idx: Int[torch.Tensor, "L"],  # noqa F821
    distance_cutoff: float,
) -> tuple[
    Float[torch.Tensor, "D len_longest_array"],
    Int[torch.Tensor, "D len_longest_array"],
    Int[torch.Tensor, "D len_longest_array"],
    Bool[torch.Tensor, "D len_longest_array"],
]:
    """
    Calculate delta distances for valid pairs from each model in batch.
    (Helper function for _calc_lddt)

    Note: This function can not be vectorized because each model in the batch may
    have different numbers of valid pairs. The returned matrices are then padded to be
    rectangular for vectorized downstream calculations.

    Parameters
    ----------
    X_L : Float[Tensor, "D L 3"]
        Predicted coordinates.
    X_gt_L : Float[Tensor, "D L 3"]
        Ground truth coordinates.
    first_index: Int[Tensor, "n_pairs"]
        First index of pairs to evaluate.
    second_index: Int[Tensor, "n_pairs"]
        Second index of pairs to evaluate.
    crd_mask_L : Bool[Tensor, "D L"]
        Coordinate mask indicating valid atoms.
    tok_idx : Int[Tensor, "L"]
        Token index of each atom. Used to exclude same-token pairs.
    distance_cutoff : float
        Distance cutoff for scoring pairs.

    Returns
    -------
    delta_distances: Float[Tensor, "D len_longest_array"]
        Pairwise distance differences (padded).
    first_index_valid: Int[Tensor, "D len_longest_array"]
        First index of pairs in delta_distances.
    second_index_valid: Int[Tensor, "D len_longest_array"]
        Second index of pairs in delta_distances.
    valid_mask: Bool[Tensor, "D len_longest_array"]
        Mask of which values in returns are valid elements (versus padding).
    """
    # Collect differently sized arrays in lists
    delta_distances_list = []
    first_index_valid_list = []
    second_index_valid_list = []

    # Calculate independently for each batch item
    for d in range(X_L.shape[0]):
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

        # Get indices only valid pairs
        valid_pairs = pair_mask.nonzero(as_tuple=True)

        # Filter ground truth distances to only valid pairs
        ground_truth_distances_valid = ground_truth_distances[valid_pairs]
        del ground_truth_distances  # since this is (n_pairs,)

        # Calculate predicted distances for only valid pairs
        first_index_valid: Int[torch.Tensor, "n_valid_pairs"] = first_index[  # noqa F821
            valid_pairs
        ]
        second_index_valid: Int[torch.Tensor, "n_valid_pairs"] = second_index[  # noqa F821
            valid_pairs
        ]
        predicted_distances = torch.linalg.norm(
            X_L[d, first_index_valid] - X_L[d, second_index_valid], dim=-1
        )

        # Calculate delta distances
        delta_distances = torch.abs(predicted_distances - ground_truth_distances_valid)
        del predicted_distances, ground_truth_distances_valid
        delta_distances_list.append(delta_distances)

        # Also keep valid pair indices (for residue-level outputs)
        first_index_valid_list.append(first_index_valid)
        second_index_valid_list.append(second_index_valid)

    # Pad into rectangular arrays
    delta_distances, valid_mask = pad_ragged(delta_distances_list)
    first_index_valid, _ = pad_ragged(first_index_valid_list)
    second_index_valid, _ = pad_ragged(second_index_valid_list)

    return delta_distances, first_index_valid, second_index_valid, valid_mask


def residue_level_lddt_single(
    threshold_passes: Int[torch.Tensor, "N"],  # noqa F821
    first_index_valid: Int[torch.Tensor, "N"],  # noqa F821
    second_index_valid: Int[torch.Tensor, "N"],  # noqa F821
    valid_mask: Bool[torch.Tensor, "N"],  # noqa F821
    tok_idx: Int[torch.Tensor, "L"],  # noqa F821
    selected_token_ids: Int[torch.Tensor, "*"],
    eps: float,
) -> Float[torch.Tensor, "n_tokens"]:  # noqa F821
    """Calculates residue-level lDDT for a single batch item
    (Helper function for _calc_lddt)

    Parameters
    ----------
    threshold_passes: Int[Tensor, "N"]
        Counts of total threshold passes for each pair.
    first_index_valid: Int[Tensor, "N"]
        First index of valid pairs.
    second_index_valid: Int[Tensor, "N"]
        Second index of valid pairs.
    tok_idx : Int[Tensor, "L"]
        Token index of each atom. Used to exclude same-token pairs.
    eps : float
        Small epsilon to prevent division by zero.
    selected_token_ids : Int[Tensor, "*"]
        Set of token IDs to compute residue-level scores for.

    Returns
    -------
    Float[torch.Tensor, "n_tokens"]
        Residue-level lDDT scores.
    """

    # Calculate for single residue
    def residue_level_lddt_single_residue(
        threshold_passes: Int[torch.Tensor, "N"],  # noqa F821
        first_index_valid: Int[torch.Tensor, "N"],  # noqa F821
        second_index_valid: Int[torch.Tensor, "N"],  # noqa F821
        valid_mask: Bool[torch.Tensor, "N"],  # noqa F821
        tok_idx: Int[torch.Tensor, "L"],  # noqa F821
        token_id: int,
        eps: float = 1e-6,
    ) -> Float[torch.Tensor, ""]:
        """Calculate residue-level lDDT for single residue

        Parameters
        ----------
        (see residue_level_lddt_single)
        token_id : int
            Token ID to compute residue-level score for.

        Returns
        -------
        torch.Float
            lDDT for only atoms in this residue.
        """
        # Filtering is simplified since same token pairs were already
        # removed upstream from delta_distances
        idxu = (tok_idx[first_index_valid] == token_id) & valid_mask  # upper triangle
        idxl = (tok_idx[second_index_valid] == token_id) & valid_mask  # lower triangle
        idx = idxu | idxl  # (N, )

        # Compute score
        return (
            0.25
            * (threshold_passes * idx).sum()  # sum over pairs involving this residue only
            / (idx.sum() + eps)
        )

    # Vectorize over all residues
    return torch.vmap(
        residue_level_lddt_single_residue,
        in_dims=(None, None, None, None, None, 0, None),
    )(
        threshold_passes,
        first_index_valid,
        second_index_valid,
        valid_mask,
        tok_idx,
        selected_token_ids,  # vectorize over tokens
        eps,
    )


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
