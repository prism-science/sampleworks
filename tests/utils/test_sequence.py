"""Tests for applying sequence conditioning to parsed structures."""

import numpy as np
import pytest
from sampleworks.utils.atom_array_utils import make_normalized_atom_id
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


def test_validate_rejects_special_characters():
    """Special characters are not valid amino acids."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        validate_seq_with_error("ACDE-FGH")


def test_validate_rejects_numeric_string():
    """Numbers are not valid amino acids."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        validate_seq_with_error("12345")


def test_validate_rejects_whitespace_embedded():
    """Whitespace within the sequence is not valid."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        validate_seq_with_error("ACD EFG")


def test_validate_accepts_canonical_amino_acids():
    """All 20 canonical amino acids should be accepted."""
    validate_seq_with_error("ACDEFGHIKLMNPQRSTVWY")


def test_override_with_no_protein_chain_raises():
    """Overriding a structure with no protein chains should raise."""
    from atomworks.enums import ChainType

    structure = {
        "chain_info": {
            "L": {
                "chain_type": ChainType.NON_POLYMER,
                "processed_entity_canonical_sequence": "",
            }
        },
        "asym_unit": None,
    }
    with pytest.raises(ValueError, match="at least one protein chain"):
        apply_sequence_override(structure, "ACDE")


def test_shorter_override_raises(structure_5i09_density: dict):
    """An override shorter than the observed sequence must fail, not silently corrupt."""
    chain_info = structure_5i09_density["chain_info"]
    protein_chain = next(cid for cid, info in chain_info.items() if info["chain_type"].is_protein())
    observed_seq = chain_info[protein_chain]["processed_entity_canonical_sequence"]
    shorter = observed_seq[: len(observed_seq) // 2]

    with pytest.raises(ValueError, match="could not be aligned"):
        apply_sequence_override(structure_5i09_density, shorter)


def test_override_with_no_chain_info_raises():
    """Overriding a structure with missing chain_info should raise."""
    with pytest.raises(ValueError, match="chain_info"):
        apply_sequence_override({}, "ACDE")


# ---------------------------------------------------------------------------
# entity_poly.seq usage: CIF with entity_poly_seq should provide the full
# deposited sequence (including unobserved residues), not just the observed
# coordinate-derived sequence. The override should replace it.
# ---------------------------------------------------------------------------


class TestEntityPolySequence:
    """Verify that atomworks-parsed CIFs use entity_poly_seq as ground truth."""

    def test_cif_with_entity_poly_populates_from_category(self, structure_1vme: dict):
        """1VME CIF has entity_poly_seq; chain_info should carry category-derived keys.

        When atomworks uses the entity_poly path, it sets
        ``unprocessed_entity_canonical_sequence`` (from entity_poly.pdbx_seq_one_letter_code).
        This key is absent when sequence is derived from coordinates alone.
        """
        chain_info = structure_1vme["chain_info"]
        for chain_id, info in chain_info.items():
            if not info["chain_type"].is_protein():
                continue
            # Marker that entity_poly was used — absent in the coordinate-only path.
            assert "unprocessed_entity_canonical_sequence" in info, (
                f"Chain {chain_id}: missing 'unprocessed_entity_canonical_sequence'; "
                f"atomworks did not use the entity_poly category"
            )
            assert len(info["processed_entity_canonical_sequence"]) > 0

    def test_cif_without_entity_poly_derives_from_coords(self, structure_5i09_density: dict):
        """5I09 density-input CIF lacks entity_poly, so sequence comes from coordinates.

        The processed_entity_canonical_sequence should match the number of unique
        observed residues, not the full deposited sequence (which is unknown to
        atomworks without the category).
        """
        chain_info = structure_5i09_density["chain_info"]
        arr = structure_5i09_density["asym_unit"]
        for chain_id, info in chain_info.items():
            if not info["chain_type"].is_protein():
                continue
            coord_seq = info["processed_entity_canonical_sequence"]
            chain_mask = np.asarray(arr.chain_id) == chain_id
            n_observed = len(np.unique(arr.res_id[chain_mask]))
            # Without entity_poly, the sequence length equals the observed count.
            assert len(coord_seq) == n_observed, (
                f"Chain {chain_id}: coordinate-derived sequence ({len(coord_seq)}) != "
                f"observed residues ({n_observed})"
            )

    def test_coord_only_cif_lacks_entity_poly_keys(self, structure_5i09_density: dict):
        """5I09 density CIF lacks entity_poly, so the category marker key should be absent."""
        chain_info = structure_5i09_density["chain_info"]
        for chain_id, info in chain_info.items():
            if not info["chain_type"].is_protein():
                continue
            assert "unprocessed_entity_canonical_sequence" not in info, (
                f"Chain {chain_id}: 'unprocessed_entity_canonical_sequence' should be "
                f"absent when entity_poly category is missing from the CIF"
            )

    def test_override_replaces_coord_derived_sequence(self, structure_5i09_density: dict):
        """Sequence override replaces the coordinate-derived sequence for 5I09."""
        chain_info = structure_5i09_density["chain_info"]
        protein_chains = [
            cid for cid, info in chain_info.items() if info["chain_type"].is_protein()
        ]
        assert len(protein_chains) == 1, "5I09 density CIF should be single-chain"

        original_seq = chain_info[protein_chains[0]]["processed_entity_canonical_sequence"]
        new_seq = "A" * (len(original_seq) + 10)
        result = apply_sequence_override(structure_5i09_density, new_seq)
        assert (
            result["chain_info"][protein_chains[0]]["processed_entity_canonical_sequence"]
            == new_seq
        )


# ---------------------------------------------------------------------------
# AtomReconciler correctness: when apply_sequence_override annotates seq_idx,
# make_normalized_atom_id must use it so the reconciler pairs atoms correctly.
# ---------------------------------------------------------------------------


class TestReconcilerWithSequenceOverride:
    """Verify that seq_idx from apply_sequence_override works with the reconciler."""

    def test_seq_idx_aligns_to_override_sequence(
        self, structure_5i09_density: dict, seq_5i09_deposited: str
    ):
        """After override with the real deposited sequence, seq_idx maps observed
        residues to their correct positions including N-terminal and internal gaps."""
        from sampleworks.eval.structure_utils import get_asym_unit_from_structure

        chain_info = structure_5i09_density["chain_info"]
        protein_chain = next(
            cid for cid, info in chain_info.items() if info["chain_type"].is_protein()
        )

        result = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)

        arr = get_asym_unit_from_structure(result, atom_array_index=0)
        chain_mask = np.asarray(arr.chain_id) == protein_chain
        seq_idx = arr.seq_idx[chain_mask]

        # All protein atoms should be mapped (>= 0).
        assert np.all(seq_idx >= 0), "All observed atoms should map to the full sequence"
        # First observed residue is auth_seq_id 10 → 0-indexed position 9 in the
        # deposited sequence (residues 1–9 are unobserved N-terminal).
        _, first_idx = np.unique(arr.res_id[chain_mask], return_index=True)
        per_residue_idx = seq_idx[np.sort(first_idx)]
        assert int(per_residue_idx.min()) >= 9, (
            "Observed residues should map after the 9-residue N-terminal gap"
        )

    def test_normalized_ids_use_seq_idx_over_dense_rank(
        self, structure_5i09_density: dict, seq_5i09_deposited: str
    ):
        """make_normalized_atom_id should use seq_idx when present, not dense rank.

        This is the core fix for B1: without seq_idx, make_normalized_atom_id
        assigns dense ranks that diverge between a gapped structure and a
        full-length model array.
        """
        # Without override: 0-based sequential.
        ids_without = make_normalized_atom_id(structure_5i09_density["asym_unit"])

        # With override using real deposited sequence: seq_idx-aware.
        result = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)
        ids_with = make_normalized_atom_id(result["asym_unit"])

        # The IDs should differ because seq_idx shifts positions for the 20 gaps.
        protein_chain = next(
            cid
            for cid, info in structure_5i09_density["chain_info"].items()
            if info["chain_type"].is_protein()
        )
        chain_mask = np.asarray(result["asym_unit"].chain_id) == protein_chain
        assert not np.array_equal(ids_without[chain_mask], ids_with[chain_mask]), (
            "make_normalized_atom_id should produce different IDs when seq_idx is present"
        )

    def test_reconciler_pairs_same_residues_after_override(
        self, structure_5i09_density: dict, seq_5i09_deposited: str
    ):
        """After override with the real deposited sequence, AtomReconciler should
        pair each atom with the correct residue — not shifted by gaps.

        Simulates the model-side array as having the full deposited sequence's
        residue range (386 residues) with dense numbering.
        """
        import biotite.structure as struc
        from sampleworks.eval.structure_utils import get_asym_unit_from_structure
        from sampleworks.utils.atom_reconciler import AtomReconciler

        chain_info = structure_5i09_density["chain_info"]
        protein_chain = next(
            cid for cid, info in chain_info.items() if info["chain_type"].is_protein()
        )

        # Apply override with the real 386-residue deposited sequence.
        struct_overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)
        struct_arr = get_asym_unit_from_structure(struct_overridden, atom_array_index=0)

        # Build a "model" array: dense residue numbering 1..386,
        # with just backbone atoms (N, CA, C, O) per residue.
        backbone = ["N", "CA", "C", "O"]
        n_res = len(seq_5i09_deposited)
        n_atoms = n_res * len(backbone)
        model_arr = struc.AtomArray(n_atoms)
        atom_names, res_ids, chain_ids = [], [], []
        for i in range(n_res):
            for atom in backbone:
                atom_names.append(atom)
                res_ids.append(i + 1)
                chain_ids.append(protein_chain)
        model_arr.atom_name = np.array(atom_names)
        model_arr.res_id = np.array(res_ids, dtype=np.int64)
        model_arr.chain_id = np.array(chain_ids)
        model_arr.coord = np.zeros((n_atoms, 3), dtype=np.float32)
        # The model array needs seq_idx too: it's dense, so seq_idx == res_id - 1.
        model_arr.set_annotation("seq_idx", np.array([r - 1 for r in res_ids], dtype=np.int64))

        # Filter struct to backbone only for comparison.
        backbone_mask = np.isin(struct_arr.atom_name, backbone)
        chain_mask = np.asarray(struct_arr.chain_id) == protein_chain
        struct_bb = struct_arr[backbone_mask & chain_mask]
        model_bb_chain = model_arr  # already single-chain backbone

        reconciler = AtomReconciler.from_arrays(model_bb_chain, struct_bb)

        # Every common atom should pair with the correct residue.
        # The struct atom at index s_idx should have the same seq_idx as model at m_idx.
        m_idx = reconciler.model_indices.numpy()
        s_idx = reconciler.struct_indices.numpy()

        model_seq_idx = model_bb_chain.seq_idx[m_idx]
        struct_seq_idx = struct_bb.seq_idx[s_idx]

        n_mispaired = int(np.sum(model_seq_idx != struct_seq_idx))
        assert n_mispaired == 0, (
            f"{n_mispaired}/{len(m_idx)} atoms mispaired after sequence override"
        )
