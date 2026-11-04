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
from sampleworks.utils.sequence import align_to_sequence


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


def _pair_input_chains(predicted_array: AtomArray, input_array: AtomArray) -> AtomArray:
    """Rename structure chains to the model chains they correspond to.

    ``make_normalized_atom_id`` pairs atoms by chain name. A structure chain pairs with the
    model chain of the same name, and remaining structure chains pair with the remaining model
    chains by position, each in order of first appearance in its atom array (a model naming
    chains A, B against an input's L, H pairs L with A).

    Parameters
    ----------
    predicted_array
        The generative model's atom array.
    input_array
        The input structure's atom array.

    Returns
    -------
    AtomArray
        ``input_array`` unchanged when every structure chain has a same-named model chain,
        otherwise a copy with renamed chains. Structure chains left over once the model's
        are used up keep their names.
    """
    predicted_chains = list(dict.fromkeys(predicted_array.chain_id.tolist()))
    input_chains = list(dict.fromkeys(input_array.chain_id.tolist()))
    unpaired_predicted = [chain for chain in predicted_chains if chain not in input_chains]
    unpaired_input = [chain for chain in input_chains if chain not in predicted_chains]
    renames = dict(zip(unpaired_input, unpaired_predicted))
    if not renames:
        return input_array
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


def _register_input_residues(predicted_array: AtomArray, input_array: AtomArray) -> AtomArray:
    """Annotate ``seq_idx`` for the structure chains with unmodeled residues.

    ``make_normalized_atom_id`` numbers residues densely per chain, so a structure chain
    with unresolved residues (e.g. a disordered terminus the model builds from the full
    sequence) is shifted against the model and pairs atoms of different residues. For each
    chain pair (with the same name) where the structure has fewer residues, the structure's
    residues are placed on the model's by name and numbering (:func:`align_to_sequence`), and
    ``seq_idx`` is set to the model position. A residue matches on any of its names, so
    microheterogeneity (two names under one number) matches either. A run of residues whose
    names match nothing but that exactly fills the space between two placed neighbours is a
    substitution (e.g. a point mutation) and takes that space, as it would in a full-length
    chain.

    Warns, without changing the pairing, about input that may pair wrongly or not at all.

    Parameters
    ----------
    predicted_array
        The generative model's atom array.
    input_array
        The input structure's atom array, chains named as their model partners.

    Returns
    -------
    AtomArray
        ``input_array`` unchanged when it already has ``seq_idx`` or no chain is shorter
        than its model chain, otherwise a copy with ``seq_idx``. Structure residues that
        align to no model residue get positions past the model chain's end, so they match
        nothing.
    """
    if "seq_idx" in input_array.get_annotation_categories():
        return input_array

    seq_idx = np.full(len(input_array), -1, dtype=np.int64)
    for chain in set(predicted_array.chain_id) & set(input_array.chain_id):
        predicted_names = _chain_residues(predicted_array[predicted_array.chain_id == chain]).names
        input_mask = input_array.chain_id == chain
        input_chain_array = input_array[input_mask]
        residue_of_atom, first_atoms, input_names, is_amino_acid, labels = _chain_residues(
            input_chain_array
        )
        non_amino_acids = [label for label, amino in zip(labels, is_amino_acid) if not amino]
        if non_amino_acids and any(is_amino_acid):
            logger.warning(
                f"Input chain {chain} mixes amino acids with other residues "
                f"{non_amino_acids[:10]} (waters or ligands sharing its chain ID). "
                "They count as residues when pairing the chain's residues with the model's."
            )
        if len(input_names) >= len(predicted_names):
            differ = [
                label
                for label, names, model in zip(labels, input_names, predicted_names)
                if not names & model
            ]
            if differ or len(input_names) > len(predicted_names):
                names_differ = f", but residue names differ at {differ[:10]}" if differ else ""
                logger.warning(
                    f"Input chain {chain} has {len(input_names)} residues to the model's "
                    f"{len(predicted_names)} and is paired with it by position{names_differ}. "
                    "If the input has residues the model lacks (e.g. an expression tag), every pair "
                    "after them is shifted."
                )
            continue
        residue_seq_idx, ambiguous_at = align_to_sequence(
            input_names,
            predicted_names,
            input_chain_array.res_id[first_atoms],
            input_chain_array.ins_code[first_atoms],
        )
        if ambiguous_at is not None:
            logger.warning(
                f"Input chain {chain} fits the model sequence in several equally good ways "
                f"around model residue index {ambiguous_at} (a gap inside a run of repeated "
                "residues that the input numbering does not place). Using one of them."
            )
        # A run of unmatched residues exactly filling the space between two matched ones is a
        # substitution: it takes the model positions between them.
        placed = np.flatnonzero(residue_seq_idx >= 0)
        for before, after in zip(placed[:-1].tolist(), placed[1:].tolist()):
            if after - before > 1 and (
                residue_seq_idx[after] - residue_seq_idx[before] == after - before
            ):
                residue_seq_idx[before + 1 : after] = residue_seq_idx[before] + np.arange(
                    1, after - before
                )
        unaligned = residue_seq_idx < 0
        if np.any(unaligned):
            unpaired = [labels[index] for index in np.flatnonzero(unaligned)]
            logger.warning(
                f"Input chain {chain} residues {unpaired[:10]} match no model residue. "
                "Their atoms are left out of the common atoms."
            )
        residue_seq_idx[unaligned] = len(predicted_names) + np.flatnonzero(unaligned)
        seq_idx[input_mask] = residue_seq_idx[residue_of_atom]

    if not np.any(seq_idx >= 0):
        return input_array
    registered = input_array.copy()
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
