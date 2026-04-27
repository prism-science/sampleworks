"""Adapter between predicted and input atom coordinate spaces.

When a generative model's atoms differ from the input structure's (atom count, residue
numbering), :class:`AtomReconciler` translates coordinates between the two.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import torch
from biotite.structure import AtomArray
from jaxtyping import Float
from loguru import logger

from sampleworks.utils.atom_array_utils import (
    closest_canonical_amino_acid,
    filter_to_common_atoms,
    residue_index,
)
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.sequence import align_sequences, align_to_sequence


def _normalize_methionine_chalcogen_names(atom_array: AtomArray) -> AtomArray:
    """Return an array with equivalent MET/MSE chalcogen names normalized.

    Parameters
    ----------
    atom_array
        Atom array whose methionine chalcogen atom names may need normalization.

    Returns
    -------
    AtomArray
        The original array when no MSE selenium is present, otherwise a copy
        where MSE ``SE`` atom names are normalized to the MET name ``SD``.
    """
    mse_selenium = (atom_array.res_name == "MSE") & (atom_array.atom_name == "SE")
    if not np.any(mse_selenium):
        return atom_array

    normalized = atom_array.copy()
    normalized.atom_name[mse_selenium] = "SD"
    return normalized


def _chains_in_order(atom_array: AtomArray) -> list[str]:
    """List an array's chain IDs in order of first appearance.

    Parameters
    ----------
    atom_array
        Atoms to read chain IDs from.

    Returns
    -------
    list[str]
        Each chain ID once.
    """
    chains, first_atoms = np.unique(atom_array.chain_id, return_index=True)
    return chains[np.argsort(first_atoms)].tolist()


def _pair_input_chains(predicted_array: AtomArray, input_array: AtomArray) -> AtomArray:
    """Rename input chains to the predicted chains they correspond to.

    Chains with the same name pair. The rest pair by how many residue names they share, most
    first; equally good partners (copies in a homo-oligomer) pair in order of appearance.

    Parameters
    ----------
    predicted_array
        The generative model's atom array.
    input_array
        The input structure's atom array.

    Returns
    -------
    AtomArray
        A copy of ``input_array`` with paired chains renamed. Chains left without a partner
        keep their names.

    Raises
    ------
    ValueError
        If an input amino-acid chain shares no residue name with any predicted chain left
        to pair with.
    """
    predicted_chains = _chains_in_order(predicted_array)
    input_chains = _chains_in_order(input_array)
    unpaired_predicted = [chain for chain in predicted_chains if chain not in input_chains]
    unpaired_input = [chain for chain in input_chains if chain not in predicted_chains]
    predicted_residues = {
        chain: _chain_residues(predicted_array[predicted_array.chain_id == chain])
        for chain in unpaired_predicted
    }
    input_residues = {
        chain: _chain_residues(input_array[input_array.chain_id == chain])
        for chain in unpaired_input
    }
    candidates = sorted(
        (
            -np.count_nonzero(align_sequences(input_res.names, predicted_res.names)[0] >= 0),
            input_order,
            predicted_order,
        )
        for input_order, (input_chain, input_res) in enumerate(input_residues.items())
        for predicted_order, predicted_res in enumerate(predicted_residues.values())
    )
    renames: dict[str, str] = {}
    for negative_shared, input_order, predicted_order in candidates:
        input_chain, predicted_chain = (
            unpaired_input[input_order],
            unpaired_predicted[predicted_order],
        )
        if (
            negative_shared < 0
            and input_chain not in renames
            and predicted_chain not in renames.values()
        ):
            renames[input_chain] = predicted_chain

    left_predicted = [chain for chain in unpaired_predicted if chain not in renames.values()]
    for chain, residues in input_residues.items():
        if chain not in renames and left_predicted and any(residues.is_amino_acid):
            raise ValueError(
                f"Input chain {chain} shares no residue name with the unpaired predicted "
                f"chains {left_predicted}, so it cannot be paired with one."
            )

    paired = input_array.copy()
    paired.chain_id = np.array([renames.get(chain, chain) for chain in paired.chain_id])
    return paired


class ChainResidues(NamedTuple):
    """A chain's atoms grouped into residues by :func:`_chain_residues`.

    Attributes
    ----------
    residue_of_atom : np.ndarray
        Residue index of each atom, shape ``(n_atoms,)``.
    first_atoms : np.ndarray
        Index of each residue's first atom, shape ``(n_residues,)``.
    names : list[frozenset[str]]
        Each residue's names, amino acids mapped to their canonical parent (``MSE -> MET``).
    is_amino_acid : list[bool]
        Whether any of each residue's names is an amino acid.
    labels : list[str]
        Each residue's names, number and insertion code for messages, e.g. ``"GLY/ALA6"``.
    """

    residue_of_atom: np.ndarray
    first_atoms: np.ndarray
    names: list[frozenset[str]]
    is_amino_acid: list[bool]
    labels: list[str]


def _chain_residues(chain_array: AtomArray) -> ChainResidues:
    """Group a chain's atoms into residues, keyed by ``(res_id, ins_code)``.

    A residue under microheterogeneity (two residue names at one position) has both names.

    Parameters
    ----------
    chain_array
        Atoms of one chain.

    Returns
    -------
    ChainResidues
    """
    residue_of_atom = residue_index(chain_array.res_id, chain_array.ins_code)
    first_atoms = np.unique(residue_of_atom, return_index=True)[1]
    raw_names: list[set[str]] = [set() for _ in first_atoms]
    for residue, name in np.unique(np.rec.fromarrays([residue_of_atom, chain_array.res_name])):
        raw_names[residue].add(str(name))
    # None for anything that is not an amino acid. Nucleic acids could get parents later.
    parents = {name: closest_canonical_amino_acid(name) for name in set().union(*raw_names)}
    names = [frozenset(parents[name] or name for name in group) for group in raw_names]
    is_amino_acid = [any(parents[name] for name in group) for group in raw_names]
    labels = [
        f"{'/'.join(sorted(group))}{chain_array.res_id[atom]}{chain_array.ins_code[atom]}"
        for group, atom in zip(raw_names, first_atoms)
    ]
    return ChainResidues(residue_of_atom, first_atoms, names, is_amino_acid, labels)


def _warn_mixed_chain(chain: str, residues: ChainResidues) -> None:
    """Warn when an amino-acid chain also holds other residues, such as waters.

    Parameters
    ----------
    chain
        Chain ID, for the message.
    residues
        The chain's residues.
    """
    others = [label for label, amino in zip(residues.labels, residues.is_amino_acid) if not amino]
    if others and any(residues.is_amino_acid):
        logger.warning(
            f"Input chain {chain} mixes amino acids with other residues {others[:10]} "
            "(waters or ligands sharing its chain ID). They count as residues when pairing "
            "the chain's residues with the predicted ones."
        )


def _place_input_residues(
    input_residues: ChainResidues,
    input_chain_array: AtomArray,
    predicted_names: list[frozenset[str]],
) -> tuple[np.ndarray, bool]:
    """Find which predicted residue each input residue corresponds to.

    If the input chain has fewer residues than the predicted chain (e.g. unresolved loops or
    termini), its residue numbers decide where the missing residues go. Otherwise the input
    may have residues the prediction lacks (e.g. an expression tag), so residues are matched
    by name alone and either chain may skip residues.

    Parameters
    ----------
    input_residues
        The input chain's residues.
    input_chain_array
        The input chain's atoms, read for their residue numbers.
    predicted_names
        Residue names of the predicted chain, in chain order.

    Returns
    -------
    positions : np.ndarray
        For each input residue, the index of the predicted residue it matched, or ``-1`` if
        it matched none. Shape ``(n_input_residues,)``.
    ambiguous : bool
        Whether a different matching is just as good, e.g. when a residue missing from a run
        of repeated residues could be any one of them.
    """
    if len(input_residues.names) >= len(predicted_names):
        return align_sequences(input_residues.names, predicted_names)
    first_atoms = input_residues.first_atoms
    positions, ambiguous_at = align_to_sequence(
        input_residues.names,
        predicted_names,
        input_chain_array.res_id[first_atoms],
        input_chain_array.ins_code[first_atoms],
    )
    return positions, ambiguous_at is not None


def _fill_substitutions(positions: np.ndarray) -> np.ndarray:
    """Match unmatched input residues that exactly fill a gap between matched neighbours.

    For example, ``[0, -1, 2]`` becomes ``[0, 1, 2]``: the middle input residue's name
    differs from predicted residue 1 (e.g. a point mutation), but it is the only residue that
    fits there. ``[0, -1, 3]`` is left as is, since one residue cannot fill a gap of two.

    Parameters
    ----------
    positions
        For each input residue, the index of its matching predicted residue, or ``-1`` if
        unmatched. Shape ``(n_residues,)``.

    Returns
    -------
    np.ndarray
        A copy with those residues matched, shape ``(n_residues,)``.
    """
    n_residues = len(positions)
    index = np.arange(n_residues)
    matched = positions >= 0
    before = np.maximum.accumulate(np.where(matched, index, 0))
    after = np.minimum.accumulate(np.where(matched, index, n_residues - 1)[::-1])[::-1]
    fits = positions[after] - positions[before] == after - before
    fill = ~matched & matched[before] & matched[after] & fits
    filled = positions.copy()
    filled[fill] = positions[before[fill]] + index[fill] - before[fill]
    return filled


def _input_chain_seq_idx(
    chain: str, predicted_chain_array: AtomArray, input_chain_array: AtomArray
) -> np.ndarray:
    """Give each atom of one input chain the index of its matching predicted residue.

    Input residues with no match get indices past the end of the predicted chain, so their
    atoms match no predicted atom.

    Parameters
    ----------
    chain
        Chain ID, for messages.
    predicted_chain_array
        The predicted chain's atoms.
    input_chain_array
        The input chain's atoms.

    Returns
    -------
    np.ndarray
        Index of the matching predicted residue for each input atom, shape
        ``(n_input_atoms,)``. All ``-1`` when the chains already match residue for residue in
        order, so residue order alone lines them up.
    """
    predicted_names = _chain_residues(predicted_chain_array).names
    input_residues = _chain_residues(input_chain_array)
    _warn_mixed_chain(chain, input_residues)
    if len(input_residues.names) == len(predicted_names) and all(
        input_res & predicted_res
        for input_res, predicted_res in zip(input_residues.names, predicted_names)
    ):
        return np.full(len(input_chain_array), -1, dtype=np.int64)

    positions, ambiguous = _place_input_residues(input_residues, input_chain_array, predicted_names)
    if ambiguous:
        logger.warning(
            f"Input chain {chain} fits the predicted chain in several equally good ways (a gap "
            "inside a run of repeated residues). Using one of them."
        )
    positions = _fill_substitutions(positions)
    unmatched = positions < 0
    if np.any(unmatched):
        labels = [input_residues.labels[index] for index in np.flatnonzero(unmatched)]
        logger.warning(
            f"Input chain {chain} residues {labels[:10]} match no predicted residue. "
            "Their atoms are left out of the common atoms."
        )
    positions[unmatched] = len(predicted_names) + np.arange(np.count_nonzero(unmatched))
    return positions[input_residues.residue_of_atom]


def _register_input_residues(predicted_array: AtomArray, input_array: AtomArray) -> AtomArray:
    """Line up the input's residues with the predicted residues they correspond to.

    Input and predicted atoms are matched by chain, residue position, and atom name
    (``make_normalized_atom_id``). By default a residue's position is its order in the chain,
    which is wrong when one side has residues the other lacks, e.g. the input is missing an
    unresolved terminus or has an expression tag the prediction lacks. Setting ``seq_idx``
    overrides that position with the index of the matching predicted residue.

    Parameters
    ----------
    predicted_array
        The generative model's atom array.
    input_array
        The input structure's atom array, with chains already renamed to their predicted
        partners.

    Returns
    -------
    AtomArray
        A copy of ``input_array`` with ``seq_idx`` set. It is not added when every chain
        already matches in order, and an existing ``seq_idx`` is kept as is.
    """
    registered = input_array.copy()
    if "seq_idx" in registered.get_annotation_categories():
        return registered
    seq_idx = np.full(len(registered), -1, dtype=np.int64)
    for chain in sorted(set(predicted_array.chain_id) & set(registered.chain_id)):
        input_mask = registered.chain_id == chain
        seq_idx[input_mask] = _input_chain_seq_idx(
            chain, predicted_array[predicted_array.chain_id == chain], registered[input_mask]
        )
    if np.any(seq_idx >= 0):
        registered.set_annotation("seq_idx", seq_idx)
    return registered


@dataclass(frozen=True, slots=True)
class AtomReconciler:
    """Bidirectional adapter between predicted (generative model) and input atom spaces.

    Computed once from the predicted and input atom arrays. Provides index-based translation
    so methods can work in predicted space and still align against the input.

    The primary method is :meth:`align`, which computes a rigid alignment on the common atom
    subset and applies it to all predicted atoms.
    Call :meth:`to` once to move index tensors to the working device before
    using the reconciler :meth:`align`.

    Attributes
    ----------
    has_mismatch : bool
        Whether reconciliation is required (count mismatch, order mismatch,
        subset mismatch, or label mismatch where chain_id, res_id, res_name,
        atom_name or element differ).
    predicted_indices : Tensor, shape ``(n_common,)``
        Indices into the predicted array for the common atom subset.
    input_indices : Tensor, shape ``(n_common,)``
        Indices into the input array for the common atom subset.
    n_predicted : int
        Total atoms in the predicted array.
    n_input : int
        Total atoms in the input array.
    n_common : int
        Atoms shared between both representations.
    """

    has_mismatch: bool
    predicted_indices: torch.Tensor
    input_indices: torch.Tensor
    n_predicted: int
    n_input: int
    n_common: int

    @classmethod
    def from_arrays(
        cls,
        predicted_array: AtomArray,
        input_array: AtomArray,
    ) -> AtomReconciler:
        """Build reconciler from predicted and input atom arrays.

        Pairs atoms by normalized atom ID after renaming input chains to their predicted
        partners (:func:`_pair_input_chains`) and lining up residues
        (:func:`_register_input_residues`).

        Parameters
        ----------
        predicted_array
            The generative model's atom array.
        input_array
            The input structure's atom array (after occupancy/NaN filtering).

        Returns
        -------
        AtomReconciler
        """
        predicted_array_for_matching = _normalize_methionine_chalcogen_names(predicted_array)
        input_array_for_matching = _register_input_residues(
            predicted_array_for_matching,
            _pair_input_chains(
                predicted_array_for_matching, _normalize_methionine_chalcogen_names(input_array)
            ),
        )
        _, (predicted_idx, input_idx) = filter_to_common_atoms(
            predicted_array_for_matching,
            input_array_for_matching,
            normalize_ids=True,
            return_indices=True,
        )

        n_predicted = len(predicted_array)
        n_input = len(input_array)
        full_coverage = n_predicted == n_input == len(predicted_idx)
        same_indexing = np.array_equal(predicted_idx, input_idx)
        identity_annotations = ("chain_id", "res_id", "res_name", "atom_name", "element")
        same_identity = all(
            np.array_equal(predicted_array.get_annotation(name), input_array.get_annotation(name))
            for name in identity_annotations
        )

        if full_coverage and same_indexing and same_identity:
            return cls.identity(n_predicted)

        return cls(
            has_mismatch=True,
            predicted_indices=torch.from_numpy(predicted_idx),
            input_indices=torch.from_numpy(input_idx),
            n_predicted=n_predicted,
            n_input=n_input,
            n_common=len(predicted_idx),
        )

    @classmethod
    def identity(
        cls,
        n_atoms: int,
    ) -> AtomReconciler:
        """No-op reconciler passthrough when atoms match.

        Parameters
        ----------
        n_atoms : int
            Number of atoms (same for predicted and input).

        Returns
        -------
        AtomReconciler
        """
        idx = torch.arange(n_atoms)
        return cls(
            has_mismatch=False,
            predicted_indices=idx,
            input_indices=idx,
            n_predicted=n_atoms,
            n_input=n_atoms,
            n_common=n_atoms,
        )

    def to(self, device: torch.device | str) -> AtomReconciler:
        """Return a copy with index tensors on the given device.

        Call once before entering a sampling loop to avoid per-step
        device transfers.

        Parameters
        ----------
        device
            Target device for index tensors.

        Returns
        -------
        AtomReconciler
            Self if already on *device*, otherwise a new instance.
        """
        device = torch.device(device) if isinstance(device, str) else device
        if self.predicted_indices.device == device:
            return self
        return replace(
            self,
            predicted_indices=self.predicted_indices.to(device),
            input_indices=self.input_indices.to(device),
        )

    def align(
        self,
        predicted_coords: Float[torch.Tensor, "*batch n_predicted 3"],
        predicted_reference: Float[torch.Tensor, "*batch n_predicted 3"],
        allow_gradients: bool = False,
        align_weights: Float[torch.Tensor, "*batch n_predicted"] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Align coords to reference using common atoms.

        Computes the rigid transform on the common atom subset, then applies
        it to all the predicted atoms.

        Parameters
        ----------
        predicted_coords
            Coordinates in predicted space, shape ``(*batch, n_predicted, 3)``.
        predicted_reference
            Reference coordinates in predicted space, shape ``(*batch, n_predicted, 3)``.
        allow_gradients
            Whether the returned coordinates stay attached to the autograd graph of
            predicted_coords. False (default) returns detached coordinates.
        align_weights
            Optional per-atom weights in predicted space, shape ``(*batch, n_predicted)``.
            Only the common-atom subset is used. Higher values increase an
            atom's influence on the rigid fit. When ``None``, all common
            atoms are weighted equally (e.g. for future confidence weighting).

        Returns
        -------
        tuple[Tensor, Mapping[str, Tensor]]
            ``(aligned_predicted_coords, transform)``
        """
        if predicted_coords.shape[-2] != self.n_predicted:
            raise ValueError(
                f"Expected predicted_coords with {self.n_predicted} atoms, "
                f"got {predicted_coords.shape[-2]}"
            )
        if predicted_reference.shape[-2] != self.n_predicted:
            raise ValueError(
                "Expected predicted_reference with "
                f"{self.n_predicted} atoms, got {predicted_reference.shape[-2]}"
            )

        # (*batch, n_common, 3)
        predicted_common = predicted_coords[..., self.predicted_indices, :]
        reference_common = predicted_reference[..., self.predicted_indices, :]

        if align_weights is not None:
            align_weights_t = torch.as_tensor(
                align_weights, device=predicted_coords.device, dtype=predicted_common.dtype
            )
            if align_weights_t.shape[-1] != self.n_predicted:
                raise ValueError(
                    f"align_weights last dimension must match n_predicted "
                    f"({self.n_predicted}); got {align_weights_t.shape[-1]}"
                )
            weights = align_weights_t[..., self.predicted_indices]
        else:
            weights = torch.ones_like(predicted_common[..., 0])

        mask = torch.ones_like(predicted_common[..., 0])
        _, transform = weighted_rigid_align_differentiable(
            predicted_common,
            reference_common,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=allow_gradients,
        )

        aligned = apply_forward_transform(predicted_coords, transform, rotation_only=False)

        # to save compute, we detach the aligned result if gradients are not allowed.
        # otherwise, the aligned result would still carry grad through the apply_forward_transform
        # call even if gradients are not needed.
        if not allow_gradients:
            aligned = aligned.detach()
        return aligned, transform

    def input_to_predicted(
        self,
        input_coords: Float[torch.Tensor, "*batch n_input 3"],
        predicted_template: Float[torch.Tensor, "*batch n_predicted 3"],
    ) -> torch.Tensor:
        """Map input atom coordinates to predicted space.

        Copies common atoms from ``input_coords`` into a clone of
        ``predicted_template``. Differentiable through ``input_coords``.

        Parameters
        ----------
        input_coords
            Coordinates in input space, shape ``(*batch, n_input, 3)``.
        predicted_template
            Template in predicted space, shape ``(*batch, n_predicted, 3)``.
            Non-common atom positions are preserved from this template.

        Returns
        -------
        Tensor
            Predicted-space coordinates, shape ``(*batch, n_predicted, 3)``.
        """
        if input_coords.shape[-2] != self.n_input:
            raise ValueError(
                f"Expected input_coords with {self.n_input} atoms, got {input_coords.shape[-2]}"
            )
        if predicted_template.shape[-2] != self.n_predicted:
            raise ValueError(
                f"Expected predicted_template with {self.n_predicted} atoms, "
                f"got {predicted_template.shape[-2]}"
            )

        result = predicted_template.clone()
        result[..., self.predicted_indices, :] = input_coords[..., self.input_indices, :]
        return result
