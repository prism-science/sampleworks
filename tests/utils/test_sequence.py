"""Tests for applying sequence conditioning to parsed structures."""

import numpy as np
import pytest
from sampleworks.utils.sequence import apply_sequence_override, validate_seq_with_error


def test_sequence_override_updates_protein_chain_without_mutating_input(structure_6b8x: dict):
    """A sequence override should update copied metadata and preserve the input."""
    protein_chain_id = next(
        chain_id
        for chain_id, info in structure_6b8x["chain_info"].items()
        if info["chain_type"].is_protein()
    )
    original_sequence = structure_6b8x["chain_info"][protein_chain_id][
        "processed_entity_canonical_sequence"
    ]

    updated = apply_sequence_override(structure_6b8x, "ACDE")

    assert updated is not structure_6b8x
    assert updated["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"] == "ACDE"
    assert (
        structure_6b8x["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"]
        == original_sequence
    )


def test_empty_sequence_preserves_structure_identity(structure_6b8x: dict):
    """Empty sequence values should retain the existing structure unchanged."""
    assert apply_sequence_override(structure_6b8x, "   ") is structure_6b8x


def test_multichain_sequence_override_raises(structure_1vme: dict):
    """Multi-chain structures should reject a single-string override."""
    with pytest.raises(ValueError, match="single-protein-chain"):
        apply_sequence_override(structure_1vme, "ACDE")


def test_longer_override_accepted_with_seq_idx(structure_6b8x: dict):
    """A longer override should be accepted and annotate atoms with seq_idx.

    The alignment maps each observed residue to its position in the full
    override sequence so the AtomReconciler pairs atoms correctly
    """
    chain_info = structure_6b8x["chain_info"]
    protein_chain_id = next(
        cid for cid, info in chain_info.items() if info["chain_type"].is_protein()
    )
    observed_seq = chain_info[protein_chain_id]["processed_entity_canonical_sequence"]
    longer = observed_seq + "GGG"

    result = apply_sequence_override(structure_6b8x, longer)

    assert result["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"] == longer
    arr = result["asym_unit"]
    assert "seq_idx" in arr.get_annotation_categories()
    chain_mask = np.asarray(arr.chain_id) == protein_chain_id
    protein_idx = arr.seq_idx[chain_mask]
    assert np.all(protein_idx >= 0), "All protein atoms should have valid seq_idx"


def test_seq_idx_not_set_on_original(structure_6b8x: dict):
    """The original atom array must not be mutated by the override."""
    chain_info = structure_6b8x["chain_info"]
    protein_chain_id = next(
        cid for cid, info in chain_info.items() if info["chain_type"].is_protein()
    )
    original_seq = chain_info[protein_chain_id]["processed_entity_canonical_sequence"]

    apply_sequence_override(structure_6b8x, original_seq + "GGG")

    assert "seq_idx" not in structure_6b8x["asym_unit"].get_annotation_categories()


def test_validate_seq_with_error_rejects_invalid_alphabet():
    """Invalid amino-acid characters should raise the shared validation error."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        validate_seq_with_error("5i09.fasta")


def test_invalid_sequence_alphabet_raises(structure_6b8x: dict):
    """Non-amino-acid characters (e.g. a mistyped path) must be rejected."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        apply_sequence_override(structure_6b8x, "5i09.fasta")

    # FIX: We should test a few other cases here, where we make sure that we catch improper inputs
