"""Translate Atomworks structures into ESMFold2 inputs."""

from typing import Any, cast

import numpy as np
import torch
from atomworks.constants import STANDARD_AA, STANDARD_AA_ONE_LETTER
from atomworks.enums import ChainType
from biotite.structure import AtomArray, AtomArrayStack, get_residue_positions, get_residue_starts
from esm.models.esmfold2.output import build_molecular_complex_from_features
from esm.models.esmfold2.prepare_input import ChainInfo
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from rdkit import rdBase


class ESMFold2InputBuilderBlockLogs(ESMFold2InputBuilder):
    """Load ESMFold2's CCD with RDKit logging temporarily silenced.

    Notes
    -----
    Our installed RDKit endlessly warns when reading ESMFold2's CCD::

        Depickling from a version number (16.3)that is higher than our version (16.2).
        This probably won't work.
        Depickling from a version number (16.3)that is higher than our version (16.2).
        This probably won't work.
        Depickling from a version number (16.3)that is higher than our version (16.2).
        This probably won't work.
        Depickling from a version number (16.3)that is higher than our version (16.2).
        This probably won't work.
        ...

    Format 16.3 requires RDKit >=2026.03.1. Our shared Atomworks 2.1.1 requires
    RDKit <2025.9, so we need Atomworks >=2.2.1 before we can upgrade RDKit
    without overriding its dependency constraint.

    Despite the warning, RDKit 2025.03.6 and 2026.03.1 loaded identical data for
    all 50,256 molecules in the current CCD and produced identical ESMFold2
    input features for a sequence containing all 20 standard amino acids.
    This wrapper suppresses RDKit logs only during initialization.

    FIXME: Remove this wrapper once Atomworks and RDKit are upgraded and replace all
    instances with ESMFold2InputBuilder.
    """

    def __init__(self, *args, **kwargs):
        """Forward builder options with RDKit logging temporarily disabled."""
        with rdBase.BlockLogs():
            super().__init__(*args, **kwargs)


def esmfold2_features_to_atom_array(
    features: dict[str, torch.Tensor], chain_infos: list[ChainInfo]
) -> AtomArray:
    """Describe the model's atoms in the order used by ESMFold2 coordinates.

    Parameters
    ----------
    features
        Batched tensors from ``ESMFold2InputBuilder.prepare_input()`` for one input.
    chain_infos
        Chain and token metadata returned alongside the features.

    Returns
    -------
    AtomArray
        Unpadded model atoms with sequential, one-based residue IDs per chain.
        Coordinates are local CCD reference positions, not an assembled structure
        or the experimental reference. Use atom identities for reconciliation.
    """
    if features["ref_pos"].shape[0] != 1:
        raise ValueError("Expected ESMFold2 features for exactly one input structure.")

    # Native decoding uses unbatched coordinates (atoms, 3) and confidence (tokens,).
    molecular_complex = build_molecular_complex_from_features(
        coords=features["ref_pos"][0].detach(),
        plddt=torch.zeros_like(features["token_attention_mask"][0], dtype=torch.float32),
        atom_mask=features["atom_attention_mask"][0],
        ref_element=features["ref_element"][0],
        ref_atom_name_chars=features["ref_atom_name_chars"][0],
        chain_infos=chain_infos,
        complex_id="model",
    )
    atoms = AtomArray(len(molecular_complex.atom_positions))
    atoms.coord = molecular_complex.atom_positions
    atoms.atom_name = np.asarray(molecular_complex.atom_names, dtype=str)
    atoms.element = np.asarray(molecular_complex.atom_elements, dtype=str)
    atoms.hetero = np.asarray(molecular_complex.atom_hetero, dtype=bool)

    residue_counts: dict[int, int] = {}
    for i, (start, stop) in enumerate(molecular_complex.token_to_atoms):
        chain_id = int(molecular_complex.chain_id[i])
        residue_counts[chain_id] = residue_counts.get(chain_id, 0) + 1
        atoms.chain_id[start:stop] = molecular_complex.metadata.chain_lookup[chain_id]
        atoms.res_id[start:stop] = residue_counts[chain_id]
        atoms.res_name[start:stop] = molecular_complex.sequence[i]

    return atoms


def structure_to_esmfold2_input(structure: dict[str, Any]) -> StructurePredictionInput:
    """Convert an Atomworks structure into an ESMFold2 input specification.

    Parameters
    ----------
    structure
        Atomworks-parsed dictionary containing ``asym_unit`` and ``chain_info``.
        For an AtomArrayStack, annotations are read from the first frame.

    Returns
    -------
    StructurePredictionInput
        ESMFold2 input specification. The original structure is not modified.
    """
    atoms = structure["asym_unit"]
    if isinstance(atoms, AtomArrayStack):
        if atoms.stack_depth() == 0:
            raise ValueError("ESMFold2 input requires a nonempty structure.")
        atoms = cast(AtomArray, atoms[0])
    if not isinstance(atoms, AtomArray):
        raise TypeError("structure['asym_unit'] must be an AtomArray or AtomArrayStack.")
    if atoms.array_length() == 0:
        raise ValueError("ESMFold2 input requires a nonempty structure.")

    chain_ids = np.unique(atoms.chain_id)
    if len(chain_ids) != 1:
        raise NotImplementedError("Multichain is unsupported.")
    chain_id = str(chain_ids[0])
    info = structure.get("chain_info", {}).get(chain_id, {})
    if info.get("chain_type") != ChainType.POLYPEPTIDE_L:
        raise NotImplementedError("Non L-protein chains are unsupported.")

    residue_codes = dict(zip(STANDARD_AA, STANDARD_AA_ONE_LETTER, strict=True))
    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    residue_names = atoms.res_name[starts[:-1]]
    unsupported = sorted(set(atoms.res_name) - residue_codes.keys())
    if unsupported:
        raise NotImplementedError(f"{unsupported} modified/nonstandard residues are unsupported.")

    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        names = atoms.atom_name[start:stop]
        if len(np.unique(names)) != len(names):
            raise ValueError(
                "Resolve alternate conformations or duplicate atoms before ESMFold2 preparation."
            )

    sequence = "".join(residue_codes[name] for name in residue_names)
    if sequence != info.get("processed_entity_canonical_sequence"):
        raise ValueError("The canonical sequence does not match the observed residues.")

    if atoms.bonds is not None:
        # Bond endpoints and their residue positions each have shape (n_bonds, 2).
        endpoints = atoms.bonds.as_array()[:, :2]
        positions = get_residue_positions(atoms, endpoints.reshape(-1)).reshape(-1, 2)
        names = atoms.atom_name[endpoints]
        forward = (
            (positions[:, 1] == positions[:, 0] + 1) & (names[:, 0] == "C") & (names[:, 1] == "N")
        )
        reverse = (
            (positions[:, 0] == positions[:, 1] + 1) & (names[:, 0] == "N") & (names[:, 1] == "C")
        )
        if np.any((positions[:, 0] != positions[:, 1]) & ~(forward | reverse)):
            raise NotImplementedError("Explicit crosslinks and cyclic peptides are unsupported.")

    return StructurePredictionInput(sequences=[ProteinInput(id=chain_id, sequence=sequence)])
