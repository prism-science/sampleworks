"""
This file originated in RosettaCommons/foundry/models/rf3 and is licensed under BSD-3-Clause.
marcus.collins@astera.org has made additions and modifications.
"""

from itertools import combinations

import numpy as np
import torch
from jaxtyping import Bool, Float
from numpy.typing import NDArray


def find_bin_midpoints(
    max_distance: float, num_bins: int, device: str | torch.device = "cpu"
) -> Float[torch.Tensor, "num_bins"]:  # noqa F821
    """Find the bin midpoints for a given binning scheme.

    Used to find expectation of values when converting binned predictions
    to unbinned predictions. Assumes the minimum of the schema is 0.

    Parameters
    ----------
    max_distance : float
        Maximum distance.
    num_bins : int
        Number of bins.
    device : str | torch.device
        Device to run on.

    Returns
    -------
    Float[Tensor, "num_bins"]
        Bin midpoints.
    """
    bin_size = max_distance / num_bins
    bins = torch.linspace(bin_size, max_distance - bin_size, num_bins - 1, device=device)
    midpoints = (bins[1:] + bins[:-1]) / 2
    midpoints = torch.cat([(bins[0] - bin_size / 2)[None], midpoints, bins[-1:] + bin_size / 2])

    return midpoints


def unbin_logits(
    logits: Float[torch.Tensor, "B num_bins L X"], max_distance: float, num_bins: int
) -> Float[torch.Tensor, "B L L"]:  # noqa F821
    """Unbin the logits to get the matrix.

    Parameters
    ----------
    logits : Float[Tensor, "B num_bins L X"]
        Binned logits where X is 23 for plddt and L for pae and pde.
    max_distance : float
        Maximum distance.
    num_bins : int
        Number of bins.

    Returns
    -------
    Float[Tensor, "B L L"]
        Unbinned matrix.
    """
    midpoints = find_bin_midpoints(max_distance, num_bins, device=logits.device)
    probabilities = torch.nn.Softmax(dim=1)(logits).detach().float()
    unbinned = (probabilities * midpoints[None, :, None, None]).sum(dim=1)
    return unbinned


def create_chainwise_masks_1d(
    ch_label: NDArray[np.str_], device: str | torch.device = "cpu"
) -> dict[str, Bool[torch.Tensor, "L"]]:  # noqa F821
    """Create 1D chainwise masks for a set of chain labels.

    Parameters
    ----------
    ch_label : NDArray[np.str_]
        Chain labels, shape (L,).
    device : str | torch.device
        Device to run on.

    Returns
    -------
    dict[str, Bool[Tensor, "L"]]
        Chain maps chain letter to which elements to score for that chain.
    """
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        indices = torch.from_numpy(ch_label == chain).to(dtype=torch.bool, device=device)
        ch_masks[chain] = indices
    return ch_masks


def create_chainwise_masks_2d(
    ch_label: NDArray[np.str_], device: str | torch.device = "cpu"
) -> dict[str, Bool[torch.Tensor, "L L"]]:  # noqa F821
    """Create 2D chainwise masks for a set of chain labels.

    Parameters
    ----------
    ch_label : NDArray[np.str_]
        Chain labels, shape (L,).
    device : str | torch.device
        Device to run on.

    Returns
    -------
    dict[str, Bool[Tensor, "L L"]]
        Chain maps chain letter to which elements to score for that chain.
    """
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        indices = torch.from_numpy(ch_label == chain)
        mask = torch.outer(indices, indices).to(dtype=torch.bool, device=device)
        ch_masks[chain] = mask
    return ch_masks


def create_interface_masks_2d(
    ch_label: NDArray[np.str_], device: str | torch.device = "cpu"
) -> dict[tuple[str, str], Bool[torch.Tensor, "L L"]]:  # noqa F821
    """Create interface masks for a set of chain labels.

    Parameters
    ----------
    ch_label : NDArray[np.str_]
        Chain labels, shape (L,).
    device : str | torch.device
        Device to run on.

    Returns
    -------
    dict[tuple[str, str], Bool[Tensor, "L L"]]
        Mapping of chain pairs to boolean masks.
    """
    unique_chains = np.unique(ch_label)
    pairs_to_score = {}
    for chain_i, chain_j in combinations(unique_chains, 2):
        chain_i_indices = torch.from_numpy(ch_label == chain_i)
        chain_j_indices = torch.from_numpy(ch_label == chain_j)
        to_be_scored = torch.outer(chain_i_indices, chain_j_indices).to(
            dtype=torch.bool, device=device
        ) + torch.outer(chain_j_indices, chain_i_indices).to(dtype=torch.bool, device=device)
        pairs_to_score[(chain_i, chain_j)] = to_be_scored
    return pairs_to_score


def compute_mean_over_subsampled_pairs(
    matrix_to_mean: Float[torch.Tensor, "B L M"],
    pairs_to_score: Bool[torch.Tensor, "L M"],
    eps: float = 1e-6,
) -> Float[torch.Tensor, "B"]:  # noqa F821
    """Compute the mean over a subsample of pairs in a 2d matrix.

    Parameters
    ----------
    matrix_to_mean : Float[Tensor, "B L M"]
        Tensor of shape (batch, L, M).
    pairs_to_score : Bool[Tensor, "L M"]
        2D tensor, shape (L, M); 1 where pairs should be scored and 0 elsewhere.
    eps : float
        Small epsilon value to avoid division by zero.

    Returns
    -------
    Float[Tensor, "B"]
        1D tensor of shape (batch,) with the mean over the subsampled pairs for each batch.
    """
    B, L, M = matrix_to_mean.shape
    assert matrix_to_mean.shape == (
        B,
        L,
        M,
    ), "Matrix to mean should be of shape (batch, L, M)"
    assert pairs_to_score.shape == (L, M), "Pairs to score should be of shape (L, M)"
    batch = (matrix_to_mean * pairs_to_score).sum(dim=(-1, -2)) / (pairs_to_score.sum() + eps)
    assert batch.shape == (B,), "Batch should be of shape (batch,)"
    return batch


def compute_min_over_subsampled_pairs(
    matrix_to_min: Float[torch.Tensor, "B L M"],  # noqa F821
    pairs_to_score: Bool[torch.Tensor, "L M"],  # noqa F821
) -> Float[torch.Tensor, "B"]:  # noqa F821
    """Compute the min over a subsample of pairs in a 2d matrix.

    Parameters
    ----------
    matrix_to_min : Float[Tensor, "B L M"]
        Tensor of shape (batch, L, M).
    pairs_to_score : Bool[Tensor, "L M"]
        2D tensor, shape (L, M); 1 where pairs should be scored and 0 elsewhere.

    Returns
    -------
    Float[Tensor, "B"]
        1D tensor of shape (batch,) with the min over the subsampled pairs for each batch.
    """
    B, L, M = matrix_to_min.shape
    assert matrix_to_min.shape == (
        B,
        L,
        M,
    ), "Matrix to min should be of shape (batch, L, M)"
    assert pairs_to_score.shape == (L, M), "Pairs to score should be of shape (L, M)"
    # Use torch.where to efficiently mask without cloning the entire matrix
    # This broadcasts pairs_to_score across the batch dimension
    masked_matrix = torch.where(
        pairs_to_score.bool(),  # condition (L, M) -> broadcasts to (B, L, M)
        matrix_to_min,  # if True: use original values (B, L, M)
        torch.tensor(
            float("inf"), device=matrix_to_min.device, dtype=matrix_to_min.dtype
        ),  # if False: use inf
    )

    # Flatten the last two dimensions and compute min across them
    batch = masked_matrix.view(B, -1).min(dim=-1)[0]

    assert batch.shape == (B,), "Batch should be of shape (batch,)"
    return batch


def spread_batch_into_dictionary(batch: Float[torch.Tensor, "B"]) -> dict[int, float]:  # noqa F821
    """Given a batch of data, create a dictionary with keys as the
    batch index and value as the corresponding data

    Parameters
    ----------
    batch : Float[Tensor, "B"]
        1D tensor of shape (B,).

    Returns
    -------
    dict[int, float]
        Dictionary mapping batch indices to float values.
    """
    assert len(batch.shape) == 1, f"Batch should be a 1d tensor, {batch}"
    return {i: data.item() for i, data in enumerate(batch)}

def pad_ragged(
    arrays: list[torch.Tensor],
    pad_value: float = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of tensors of different lengths, pad to the length
    of the longest tensor in order to construct a rectangular array.

    Parameters
    ----------
    arrays: list[Tensor]
        Python list of 1D tensors of varying lengths
    pad_value: float
        Value to pad arrays with.
    
    Returns
    -------
    padded: Tensor
        Padded rectangular tensor of shape [len(arrays), len(longest array)] 
    valid_mask: Tensor
        Mask indicating which padded_array values are valid (versus padded).
    """
    # Determine length to pad to
    assert len(arrays) > 0, "No elements were provided"
    max_len = max(len(array) for array in arrays)

    # Pad array
    assert all(array.dtype == arrays[0].dtype for array in arrays), (
        "Input arrays have different dtype"
    )
    assert all(array.device == arrays[0].device for array in arrays), (
        "Input arrays are placed on different devices"
    )
    padded = torch.full(
        (len(arrays), max_len),
        pad_value,
        dtype=arrays[0].dtype,
        device=arrays[0].device
    )

    # Track which elements are valid
    valid_mask = torch.zeros(
        len(arrays), max_len,
        dtype=torch.bool,
        device=arrays[0].device
    )

    # Fill in real values
    for i, array in enumerate(arrays):
        n = len(array)
        padded[i, :n] = array
        valid_mask[i, :n] = True

    return padded, valid_mask