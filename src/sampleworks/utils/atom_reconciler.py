"""Adapter between model and structure atom coordinate spaces.

When a generative model's internal atom representation differs from the input
structure (different atom count, different residue numbering), the
:class:`AtomReconciler` translates coordinates between the two spaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from biotite.structure import AtomArray
from jaxtyping import Float

from sampleworks.utils.atom_array_utils import filter_to_common_atoms
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)


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


@dataclass(frozen=True, slots=True)
class AtomReconciler:
    """Bidirectional adapter between model and structure atom spaces.

    Computed once from the model's and structure's atom arrays. Provides
    index-based translation so methods can work in model space and still
    do things like alignment against the structure.

    The primary method is :meth:`align`, which computes a rigid
    alignment on the common atom subset between the input structure atoms and the model-derived
    atoms and applies it to all model atoms.
    Call :meth:`to` once to move index tensors to the working device before
    using the reconciler :meth:`align`.

    Attributes
    ----------
    has_mismatch : bool
        Whether reconciliation is required (count mismatch, order mismatch,
        or subset mismatch).
    model_indices : Tensor, shape ``(n_common,)``
        Indices into the model array for the common atom subset.
    struct_indices : Tensor, shape ``(n_common,)``
        Indices into the structure array for the common atom subset.
    n_model : int
        Total atoms in the model representation.
    n_struct : int
        Total atoms in the structure representation.
    n_common : int
        Atoms shared between both representations.
    mapped_struct_mask : Tensor, shape ``(n_struct,)``
        Boolean mask selecting structure atoms backed by model coordinates.
    """

    has_mismatch: bool
    model_indices: torch.Tensor
    struct_indices: torch.Tensor
    n_model: int
    n_struct: int
    n_common: int

    @property
    def mapped_struct_mask(self) -> torch.Tensor:
        """Return the structure atoms backed by mapped model coordinates.

        Returns
        -------
        Tensor
            Boolean mask with shape ``(n_struct,)`` on the index tensor device.
        """
        mask = torch.zeros(self.n_struct, dtype=torch.bool, device=self.struct_indices.device)
        mask[self.struct_indices] = True
        return mask

    @classmethod
    def from_arrays(
        cls,
        model_array: AtomArray,
        struct_array: AtomArray,
    ) -> AtomReconciler:
        """Build reconciler from model and structure atom arrays.

        Uses normalized atom IDs (sequential per-chain residue numbering)
        to handle numbering differences between representations.

        Parameters
        ----------
        model_array
            The generative model's internal atom array.
        struct_array
            The input structure's atom array (after occupancy/NaN filtering).

        Returns
        -------
        AtomReconciler
        """
        model_array_for_matching = _normalize_methionine_chalcogen_names(model_array)
        struct_array_for_matching = _normalize_methionine_chalcogen_names(struct_array)
        _, (m_idx, s_idx) = filter_to_common_atoms(
            model_array_for_matching,
            struct_array_for_matching,
            normalize_ids=True,
            return_indices=True,
        )

        n_model = len(model_array)
        n_struct = len(struct_array)
        full_coverage = n_model == n_struct == len(m_idx)
        same_indexing = np.array_equal(m_idx, s_idx)
        identity_annotations = ("chain_id", "res_id", "res_name", "atom_name", "element")
        same_identity = all(
            np.array_equal(model_array.get_annotation(name), struct_array.get_annotation(name))
            for name in identity_annotations
        )

        if full_coverage and same_indexing and same_identity:
            return cls.identity(n_model)

        return cls(
            has_mismatch=True,
            model_indices=torch.from_numpy(m_idx),
            struct_indices=torch.from_numpy(s_idx),
            n_model=n_model,
            n_struct=n_struct,
            n_common=len(m_idx),
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
            Number of atoms (same for model and structure).

        Returns
        -------
        AtomReconciler
        """
        idx = torch.arange(n_atoms)
        return cls(
            has_mismatch=False,
            model_indices=idx,
            struct_indices=idx,
            n_model=n_atoms,
            n_struct=n_atoms,
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
        if self.model_indices.device == device:
            return self
        return replace(
            self,
            model_indices=self.model_indices.to(device),
            struct_indices=self.struct_indices.to(device),
        )

    def align(
        self,
        model_coords: Float[torch.Tensor, "*batch n_model 3"],
        model_reference: Float[torch.Tensor, "*batch n_model 3"],
        allow_gradients: bool = False,
        align_weights: Float[torch.Tensor, "*batch n_model"] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Align coords to reference using common atoms.

        Computes the rigid transform on the common atom subset, then applies
        it to all the model atoms.

        Parameters
        ----------
        model_coords
            Coordinates in model space, shape ``(*batch, n_model, 3)``.
        model_reference
            Reference coordinates in model space, shape ``(*batch, n_model, 3)``.
        allow_gradients
            Whether to preserve gradients through alignment.
        align_weights
            Optional per-atom weights in model space, shape ``(*batch, n_model)``.
            Only the common-atom subset is used. Higher values increase an
            atom's influence on the rigid fit. When ``None``, all common
            atoms are weighted equally (e.g. for future confidence weighting).

        Returns
        -------
        tuple[Tensor, Mapping[str, Tensor]]
            ``(aligned_model_coords, transform)``
        """
        if model_coords.shape[-2] != self.n_model:
            raise ValueError(
                f"Expected model_coords with {self.n_model} atoms, got {model_coords.shape[-2]}"
            )
        if model_reference.shape[-2] != self.n_model:
            raise ValueError(
                "Expected model_reference with "
                f"{self.n_model} atoms, got {model_reference.shape[-2]}"
            )

        # (*batch, n_common, 3)
        model_common = model_coords[..., self.model_indices, :]
        reference_common = model_reference[..., self.model_indices, :]

        if align_weights is not None:
            align_weights_t = torch.as_tensor(
                align_weights, device=model_coords.device, dtype=model_common.dtype
            )
            if align_weights_t.shape[-1] != self.n_model:
                raise ValueError(
                    f"align_weights last dimension must match n_model "
                    f"({self.n_model}); got {align_weights_t.shape[-1]}"
                )
            weights = align_weights_t[..., self.model_indices]
        else:
            weights = torch.ones_like(model_common[..., 0])

        mask = torch.ones_like(model_common[..., 0])
        _, transform = weighted_rigid_align_differentiable(
            model_common,
            reference_common,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=allow_gradients,
        )

        aligned = apply_forward_transform(model_coords, transform, rotation_only=False)
        return aligned, transform

    def struct_to_model(
        self,
        struct_coords: Float[torch.Tensor, "*batch n_struct 3"],
        model_template: Float[torch.Tensor, "*batch n_model 3"],
    ) -> torch.Tensor:
        """Map structure atom coordinates to model.

        Copies common atoms from ``struct_coords`` into a clone of
        ``model_template``. Differentiable through ``struct_coords``.

        Parameters
        ----------
        struct_coords
            Coordinates in structure space, shape ``(*batch, n_struct, 3)``.
        model_template
            Template in model space, shape ``(*batch, n_model, 3)``.
            Non-common atom positions are preserved from this template.

        Returns
        -------
        Tensor
            Model-space coordinates, shape ``(*batch, n_model, 3)``.
        """
        if struct_coords.shape[-2] != self.n_struct:
            raise ValueError(
                f"Expected struct_coords with {self.n_struct} atoms, got {struct_coords.shape[-2]}"
            )
        if model_template.shape[-2] != self.n_model:
            raise ValueError(
                f"Expected model_template with {self.n_model} atoms, got {model_template.shape[-2]}"
            )

        result = model_template.clone()
        result[..., self.model_indices, :] = struct_coords[..., self.struct_indices, :]
        return result

    def model_to_struct(
        self,
        model_coords: Float[torch.Tensor, "*batch n_model 3"],
        struct_template: Float[torch.Tensor, "*batch n_struct 3"],
    ) -> torch.Tensor:
        """Map model atom coordinates to structure space.

        Copies mapped atoms from ``model_coords`` into a clone of
        ``struct_template``. Differentiable through ``model_coords``. Use
        :attr:`mapped_struct_mask` to exclude unmapped template coordinates.

        Parameters
        ----------
        model_coords
            Coordinates in model space, shape ``(*batch, n_model, 3)``.
        struct_template
            Template in structure space, shape ``(*batch, n_struct, 3)``.

        Returns
        -------
        Tensor
            Structure-space coordinates, shape ``(*batch, n_struct, 3)``.
        """
        if model_coords.shape[-2] != self.n_model:
            raise ValueError(
                f"Expected model_coords with {self.n_model} atoms, got {model_coords.shape[-2]}"
            )
        if struct_template.shape[-2] != self.n_struct:
            raise ValueError(
                f"Expected struct_template with {self.n_struct} atoms, "
                f"got {struct_template.shape[-2]}"
            )

        result = struct_template.clone()
        result[..., self.struct_indices, :] = model_coords[..., self.model_indices, :]
        return result
