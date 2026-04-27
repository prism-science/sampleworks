"""Tests for applying sequence conditioning to parsed structures."""

from pathlib import Path

import numpy as np
import pytest
from sampleworks.utils.atom_array_utils import make_normalized_atom_id, parse_structure
from sampleworks.utils.sequence import (
    align_sequences,
    align_to_sequence,
    apply_sequence_override,
    infer_sequence_override,
    validate_seq_with_error,
)


@pytest.mark.parametrize("stacked", [False, True])
def test_override_preserves_insertion_code_residues(stacked):
    """Map insertion-code residues independently, retaining all coordinate frames.

    Parameters
    ----------
    stacked : bool
        Whether to supply a multi-model structure with shared annotations.
    """
    from atomworks.enums import ChainType
    from biotite.structure import AtomArray, stack

    atoms = AtomArray(6)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 10
    atoms.ins_code = np.repeat(["", "A", "B"], 2)
    atoms.res_name = np.repeat(["ALA", "CYS", "ASP"], 2)
    atoms.atom_name = np.tile(["N", "CA"], 3)
    atoms.coord = np.arange(18, dtype=np.float32).reshape(6, 3)
    array = stack([atoms, atoms.copy()]) if stacked else atoms
    structure = {
        "asym_unit": array,
        "chain_info": {
            "A": {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "processed_entity_canonical_sequence": "ACD",
            }
        },
    }

    updated = apply_sequence_override(structure, "MACDE")["asym_unit"]

    np.testing.assert_array_equal(updated.seq_idx, [1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(updated.coord, array.coord)
    assert len(np.unique(make_normalized_atom_id(updated))) == 6
    assert "seq_idx" not in array.get_annotation_categories()


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

    override = original_sequence + "GGG"
    updated = apply_sequence_override(structure_6b8x, override)

    assert updated is not structure_6b8x
    assert (
        updated["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"] == override
    )
    assert (
        structure_6b8x["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"]
        == original_sequence
    )


@pytest.mark.parametrize("res_name, sequence", [("MSE", "M"), ("CSO", "C"), ("ASP", "E")])
def test_override_requires_matching_canonical_residue(res_name, sequence):
    """Accept canonicalized modifications but reject residue substitutions.

    Parameters
    ----------
    res_name : str
        Observed CCD residue name.
    sequence : str
        Candidate one-residue override.
    """
    from atomworks.enums import ChainType
    from biotite.structure import AtomArray

    atoms = AtomArray(1)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.res_name[:] = res_name
    atoms.atom_name[:] = "CA"
    structure = {
        "asym_unit": atoms,
        "chain_info": {"A": {"chain_type": ChainType.POLYPEPTIDE_L}},
    }

    if res_name == "ASP":
        with pytest.raises(ValueError, match="could not be aligned"):
            apply_sequence_override(structure, sequence)
    else:
        updated = apply_sequence_override(structure, sequence)
        np.testing.assert_array_equal(updated["asym_unit"].seq_idx, [0])
        np.testing.assert_array_equal(updated["asym_unit"].res_name, [res_name])


def test_empty_sequence_preserves_structure_identity(structure_6b8x: dict):
    """Empty sequence values should retain the existing structure unchanged."""
    assert apply_sequence_override(structure_6b8x, "   ") is structure_6b8x


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


@pytest.mark.parametrize("sequence", ["ACDX", "ACDB", "ACDZ", "ACDU", "ACD*"])
def test_validate_rejects_non_canonical_codes(sequence: str):
    """Ambiguous, rare, and stop codes are rejected rather than reaching CCD lookups."""
    with pytest.raises(ValueError, match="Invalid protein sequence"):
        validate_seq_with_error(sequence)


def test_validate_returns_uppercased_sequence():
    """Lowercase input is accepted and returned in canonical uppercase form."""
    assert validate_seq_with_error("acdefghiklmnpqrstvwy") == "ACDEFGHIKLMNPQRSTVWY"


def test_lowercase_override_aligns_like_uppercase(structure_6b8x: dict):
    """A lowercase override must store the uppercase sequence and map every residue."""
    protein_chain_id, info = next(
        (cid, info)
        for cid, info in structure_6b8x["chain_info"].items()
        if info["chain_type"].is_protein()
    )
    longer = info["processed_entity_canonical_sequence"] + "GGG"

    result = apply_sequence_override(structure_6b8x, longer.lower())

    assert result["chain_info"][protein_chain_id]["processed_entity_canonical_sequence"] == longer
    arr = result["asym_unit"]
    assert np.all(arr.seq_idx[np.asarray(arr.chain_id) == protein_chain_id] >= 0)


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
        new_seq = original_seq + "A" * 10
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
        from sampleworks.utils.structure_utils import get_asym_unit_from_structure

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
        """make_normalized_atom_id numbers residues by seq_idx when it is set.

        Counting observed residues 0, 1, 2, ... ignores unresolved residues, so the numbers
        drift from a full-length predicted chain; seq_idx gives each residue its position in
        the deposited sequence instead.
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
        arr = result["asym_unit"]
        chain_mask = np.asarray(arr.chain_id) == protein_chain
        assert not np.array_equal(ids_without[chain_mask], ids_with[chain_mask]), (
            "make_normalized_atom_id should produce different IDs when seq_idx is present"
        )

        # 5I09 author numbering is deposited position + 1, so each protein atom's
        # normalized position must be its res_id - 1.
        protein = arr.seq_idx >= 0
        protein_ids = ids_with[protein]
        np.testing.assert_array_equal(protein_ids["chain_id"], arr.chain_id[protein])
        np.testing.assert_array_equal(protein_ids["seq_pos"], arr.res_id[protein] - 1)
        np.testing.assert_array_equal(protein_ids["atom_name"], arr.atom_name[protein])

    def test_reconciler_pairs_same_residues_after_override(
        self, structure_5i09_density: dict, seq_5i09_deposited: str
    ):
        """After override with the real deposited sequence, AtomReconciler should
        pair each atom with the correct residue — not shifted by gaps.

        Simulates the model-side array the way a wrapper emits it: the full deposited
        sequence (386 residues) with dense numbering and no ``seq_idx``. 5I09 author
        numbering is deposited position + 1, so model residue k must pair with
        structure residue k.
        """
        struct_overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)

        _assert_backbone_pairs_same_residues(struct_overridden, seq_5i09_deposited)


_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def _backbone_model_array(sequence: str, chain_id: str):
    """Backbone atoms of ``sequence`` numbered 1..N on one chain, as a model wrapper emits them."""
    from biotite.sequence import ProteinSequence
    from biotite.structure import AtomArray

    n_atoms_per_residue = len(_BACKBONE_ATOM_NAMES)
    model_arr = AtomArray(len(sequence) * n_atoms_per_residue)
    model_arr.atom_name = np.tile(_BACKBONE_ATOM_NAMES, len(sequence))
    model_arr.res_id = np.repeat(np.arange(1, len(sequence) + 1), n_atoms_per_residue)
    model_arr.res_name = np.repeat(
        [ProteinSequence.convert_letter_1to3(aa) for aa in sequence], n_atoms_per_residue
    )
    model_arr.chain_id[:] = chain_id
    return model_arr


def _assert_backbone_pairs_same_residues(structure: dict, model_sequence: str) -> None:
    """Assert the reconciler pairs every input backbone atom with the same model residue.

    The model array is ``model_sequence`` numbered from 1, so it is the same residue
    as an input residue with the same ``res_id`` when the input is numbered by
    sequence position (``label_seq_id`` or 5I09's author numbering).
    """
    from sampleworks.utils.atom_reconciler import AtomReconciler
    from sampleworks.utils.structure_utils import get_asym_unit_from_structure

    protein_chain = next(
        cid for cid, info in structure["chain_info"].items() if info["chain_type"].is_protein()
    )
    struct_arr = get_asym_unit_from_structure(structure, atom_array_index=0)
    struct_bb = struct_arr[
        np.isin(struct_arr.atom_name, _BACKBONE_ATOM_NAMES)
        & (np.asarray(struct_arr.chain_id) == protein_chain)
    ]
    model_arr = _backbone_model_array(model_sequence, protein_chain)

    reconciler = AtomReconciler.from_arrays(model_arr, struct_bb)
    m_idx = reconciler.predicted_indices.numpy()  # (n_common,)
    s_idx = reconciler.input_indices.numpy()  # (n_common,)

    assert len(s_idx) == struct_bb.array_length()
    np.testing.assert_array_equal(model_arr.res_id[m_idx], struct_bb.res_id[s_idx])
    np.testing.assert_array_equal(model_arr.res_name[m_idx], struct_bb.res_name[s_idx])
    np.testing.assert_array_equal(model_arr.atom_name[m_idx], struct_bb.atom_name[s_idx])


# ---------------------------------------------------------------------------
# Inferring the override from the sequence the structure file states (entity_poly)
# ---------------------------------------------------------------------------


def test_inferred_override_pairs_model_with_unresolved_residues(resources_dir: Path):
    """Without --sequence, a CIF's entity_poly sequence still pairs atoms across its gaps.

    Loaded as guidance loads it, without atomworks adding the unresolved residues back.
    """
    structure = parse_structure(resources_dir / "5I09" / "5I09_prediction.cif")
    stated_sequence = structure["chain_info"]["A"]["processed_entity_canonical_sequence"]

    overridden = apply_sequence_override(structure, infer_sequence_override(structure))

    _assert_backbone_pairs_same_residues(overridden, stated_sequence)


def test_no_inferred_override_without_entity_poly(structure_5i09_density: dict):
    """A sequence read off the coordinates cannot reveal unresolved residues."""
    assert infer_sequence_override(structure_5i09_density) is None


def test_no_inferred_override_without_unresolved_residues():
    """A stated sequence fully covered by resolved residues needs no override."""
    structure = _single_chain_structure([1, 2, 3], "MKL")
    structure["chain_info"]["A"]["unprocessed_entity_canonical_sequence"] = "MKL"

    assert infer_sequence_override(structure) is None


def test_no_inferred_override_for_multiple_protein_chains(resources_dir: Path):
    """Inference covers single-protein-chain structures only, so multi-chain inputs get none."""
    structure = parse_structure(resources_dir / "1vme" / "1vme_final.cif")

    assert infer_sequence_override(structure) is None


# ---------------------------------------------------------------------------
# Gap placement: runs of consecutive residue numbers must stay contiguous
# ---------------------------------------------------------------------------

# RCSB 5I09 chain A entity sequence (pdbx_seq_one_letter_code_can). Author number = position + 1.
_5I09_FULL_SEQUENCE = (
    "MVEATAQETDRPRFSFSIAAREGKARTGTIEMKRGVIRTPAFMPVGTAATVKALKPETVRATGADIILGNTYHLMLRPGA"
    "ERIAKLGGLHSFMGWDRPILTDSGGYQVMSLSSLTKQSEEGVTFKSHLDGSRHMLSPERSIEIQHLLGSDIVMAFDECTP"
    "YPATPSRAASSMERSMRWAKRSRDAFDSRKEQAENAALFGIQQGSVFENLRQQSADALAEIGFDGYAVGGLAVGEGQDEM"
    "FRVLDFSVPMLPDDKPHYLMGVGKPDDIVGAVERGIDMFDCVLPTRSGRNGQAFTWDGPINIRNARFSEDLKPLDSECHC"
    "AVCQKWSRAYIHHLIRAGEILGAMLMTEHNIAFYQQLMQKIRDSISEGRFSQFAQDFRARYFARNS"
)


def _single_chain_structure(
    res_ids: list[int], one_letter: str, ins_codes: list[str] | None = None
) -> dict:
    """One CA atom per residue on chain A, with the observed sequence as chain_info."""
    from atomworks.enums import ChainType
    from biotite.sequence import ProteinSequence
    from biotite.structure import AtomArray

    atoms = AtomArray(len(res_ids))
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = res_ids
    atoms.res_name = np.array([ProteinSequence.convert_letter_1to3(c) for c in one_letter])
    atoms.atom_name[:] = "CA"
    if ins_codes is not None:
        atoms.ins_code = np.array(ins_codes)
    return {
        "asym_unit": atoms,
        "chain_info": {
            "A": {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "processed_entity_canonical_sequence": one_letter,
            }
        },
    }


def test_residue_after_gap_is_not_pulled_across_the_gap():
    """A residue that also occurs just before a gap must have the correct numbering.

    Full sequence ``MVEATAQETDFKSHLDGSRHMLS`` is numbered from 114 (the 5I09 motif around
    its 126-130 gap). With ``SHLDG`` missing, a sequence-only alignment starts the observed
    ``SRHMLS`` at SER 126, right after LYS 125, and then cannot place the rest of the sequence
    properly, but the residue numbers say the run starts at SER 131 so we should use that info.
    """
    res_ids = [*range(114, 126), *range(131, 137)]
    structure = _single_chain_structure(res_ids, "MVEATAQETDFK" + "SRHMLS")

    updated = apply_sequence_override(structure, "MVEATAQETDFKSHLDGSRHMLS")["asym_unit"]

    np.testing.assert_array_equal(updated.seq_idx, [*range(0, 12), *range(17, 23)])


def test_5i09_observed_residues_follow_author_numbering(structure_5i09_density: dict):
    """Regression for 5I09: every observed residue lands at author number - 1 (RCSB numbering)."""
    updated = apply_sequence_override(structure_5i09_density, _5I09_FULL_SEQUENCE)["asym_unit"]

    protein = updated.seq_idx >= 0
    np.testing.assert_array_equal(updated.res_id[protein] - updated.seq_idx[protein], 1)


@pytest.mark.parametrize(
    ("full_sequence", "observed_positions"),
    [
        ("AAAAAA", [0, 1, 4, 5]),
        # This example previously had a bug where the alignment offset was not applied correctly
        # since the gap penalty is 0 at the ends:
        # full:      I S M C G R I H C K N L F M P     (positions 0–14)
        # observed:  I S _ C _ _ _ _ _ K N L F M P     (res_ids 1,2,4,10–15 -> should be aligned to
        # true positions 0,1,3,9–14)
        ("ISMCGRIHCKNLFMP", [0, 1, 3, 9, 10, 11, 12, 13, 14]),
    ],
)
def test_ambiguous_alignment_follows_consistent_numbering(full_sequence, observed_positions):
    """When the alignment ties, residue numbers (res_id = position + 1) decide the placement."""
    observed = "".join(full_sequence[p] for p in observed_positions)
    structure = _single_chain_structure([p + 1 for p in observed_positions], observed)

    updated = apply_sequence_override(structure, full_sequence)["asym_unit"]

    np.testing.assert_array_equal(updated.seq_idx, observed_positions)


def test_insertion_codes_next_to_gap():
    """Insertion-code residues share a res_id but still take consecutive sequence positions."""
    structure = _single_chain_structure(
        [1, 2, 3, 3, 3, 8, 9], "MACDEHI", ins_codes=["", "", "", "A", "B", "", ""]
    )

    updated = apply_sequence_override(structure, "MACDEFGHI")["asym_unit"]

    np.testing.assert_array_equal(updated.seq_idx, [0, 1, 2, 3, 4, 7, 8])


def test_override_warns_when_numbering_does_not_place_a_gap(caplog):
    """A His-tag renumbered contiguously with one HIS missing fits several ways; say so."""
    structure = _single_chain_structure([1, 2, 3, 4, 5], "MHHHG")

    apply_sequence_override(structure, "MHHHHG")

    assert "several equally good ways" in caplog.text


def _align(observed, sequence, numbers=None, codes=None):
    """Run align_to_sequence on lists, returning positions as a list and the ambiguity."""
    positions, ambiguous_at = align_to_sequence(
        observed,
        sequence,
        None if numbers is None else np.array(numbers),
        None if numbers is None else np.array(codes or [""] * len(numbers)),
    )
    return positions.tolist(), ambiguous_at


def test_align_to_sequence_without_numbering():
    """Exact matches, the fewest skipped runs, alternatives, ties where they start, no match."""
    assert _align(["GLY", "HIS", "SER"], ["GLY", "HIS", "SER"]) == ([0, 1, 2], None)
    # One run of four skipped positions beats a skipped run at each end.
    assert _align(["ALA", "ALA"], ["ALA", "GLY", "SER", "ALA", "THR", "ALA"]) == ([0, 5], None)
    # A microheterogeneity position matches any of its monomers.
    assert _align(["ASP", "HIS"], ["GLY", {"ASN", "ASP"}, "HIS"]) == ([1, 2], None)
    # Two of three alanines fit three ways, disagreeing from the first position.
    assert _align(["ALA", "ALA"], ["ALA"] * 3)[1] == 0
    # A tie is reported where it starts, not at the start of the sequence.
    assert _align(["GLY", "HIS", "ALA", "ALA"], ["GLY", "HIS", "ALA", "ALA", "ALA"])[1] == 2
    assert _align(["GLY"], ["ALA"]) == ([-1], None)


def test_align_to_sequence_places_gaps_where_numbering_skips():
    """Numbering places a gap inside a run of identical residues, where names alone cannot."""
    # His-tag with residue 4 unresolved: by name alone the gap could be after any HIS.
    sequence = list("MHHHHHHG")
    observed, numbers = list("MHHHHHG"), [1, 2, 3, 5, 6, 7, 8]

    assert _align(observed, sequence, numbers) == ([0, 1, 2, 4, 5, 6, 7], None)
    assert _align(observed, sequence)[1] is not None


def test_align_to_sequence_matches_any_observed_monomer():
    """An observed microheterogeneity residue matches a position holding any of its monomers."""
    assert _align([{"ALA", "SER"}, "HIS"], ["SER", "HIS"]) == ([0, 1], None)
    assert _align([{"ALA", "SER"}, "HIS"], ["GLY", "HIS"]) == ([-1, 1], None)


def test_align_to_sequence_keeps_mismatches_in_place():
    """A residue whose name matches nowhere it fits stays unplaced instead of shifting others."""
    assert _align(list("MKXEF"), list("MKLEF"), [1, 2, 3, 4, 5]) == ([0, 1, -1, 3, 4], None)


def test_reset_numbering_raises_malformed():
    """A chain whose numbering restarts raises a malformed error."""
    structure = _single_chain_structure([1, 2, 3, 1, 2], "MKLMK")

    with pytest.raises(ValueError, match="malformed"):
        apply_sequence_override(structure, "MKLMK")


def test_repeated_residue_number_raises_malformed():
    """Two residues under one number (ASN/ASP at 3) cannot each take an override position."""
    structure = _single_chain_structure([1, 2, 3, 3, 4], "MKNDE")

    with pytest.raises(ValueError, match="malformed"):
        apply_sequence_override(structure, "MKNDEF")


def test_ligand_in_protein_chain_raises_malformed(structure_1vme_density_input: dict):
    """Ligands deposited inside the protein chain are reported and not silently misaligned."""
    chain_a = structure_1vme_density_input["chain_info"]["A"]
    sequence = chain_a["processed_entity_canonical_sequence"]

    with pytest.raises(ValueError, match="malformed") as excinfo:
        apply_sequence_override(structure_1vme_density_input, sequence.replace("X", ""))

    assert "FEO" in str(excinfo.value)


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (["SER", "GLY", "HIS"], ["GLY", "HIS", "TRP"], ([-1, 0, 1], False)),
        (["GLY", "ALA", "HIS"], ["GLY", "SER", "HIS"], ([0, -1, 2], False)),
        (["GLY", {"ALA", "SER"}], ["GLY", "SER"], ([0, 1], False)),
        (["GLY", "HIS", "SER"], ["GLY", "HIS", "HIS", "SER"], ([0, 1, 3], True)),
    ],
    ids=["extra_residue_on_each_side", "substitution", "either_name", "gap_in_a_repeat"],
)
def test_align_sequences_skips_residues_on_either_side(first, second, expected):
    positions, ambiguous = align_sequences(first, second)

    assert (positions.tolist(), ambiguous) == expected


# ---------------------------------------------------------------------------
# Multi-chain overrides: one ":"-separated sequence per protein chain
# ---------------------------------------------------------------------------


def _two_chain_structure(chain_a: str, chain_b: str) -> dict:
    """One CA atom per residue on protein chains A and B, numbered from 1."""
    from atomworks.enums import ChainType
    from biotite.sequence import ProteinSequence
    from biotite.structure import AtomArray

    one_letter = chain_a + chain_b
    atoms = AtomArray(len(one_letter))
    atoms.chain_id = np.array(["A"] * len(chain_a) + ["B"] * len(chain_b))
    atoms.res_id = np.array([*range(1, len(chain_a) + 1), *range(1, len(chain_b) + 1)])
    atoms.res_name = np.array([ProteinSequence.convert_letter_1to3(c) for c in one_letter])
    atoms.atom_name[:] = "CA"
    return {
        "asym_unit": atoms,
        "chain_info": {
            chain_id: {
                "chain_type": ChainType.POLYPEPTIDE_L,
                "processed_entity_canonical_sequence": sequence,
            }
            for chain_id, sequence in (("A", chain_a), ("B", chain_b))
        },
    }


def test_multichain_override_assigns_sequences_in_chain_order():
    """The n-th ``:``-separated sequence overrides the n-th protein chain."""
    structure = _two_chain_structure("MKL", "WYF")

    updated = apply_sequence_override(structure, "MKLG:WYFHH")

    chain_info = updated["chain_info"]
    assert chain_info["A"]["processed_entity_canonical_sequence"] == "MKLG"
    assert chain_info["B"]["processed_entity_canonical_sequence"] == "WYFHH"
    np.testing.assert_array_equal(updated["asym_unit"].seq_idx, [0, 1, 2, 0, 1, 2])


def test_multichain_override_in_wrong_chain_order_raises():
    """Sequences that do not match their chain's residues fail to align."""
    with pytest.raises(ValueError, match="could not be aligned"):
        apply_sequence_override(_two_chain_structure("MKL", "WYF"), "WYFG:MKLG")


@pytest.mark.parametrize("sequence", ["ACDE", "ACDE:ACDE:ACDE"])
def test_multichain_override_with_wrong_chain_count_raises(structure_1vme: dict, sequence: str):
    """The number of sequences must equal the number of protein chains."""
    with pytest.raises(ValueError, match="2 protein chains"):
        apply_sequence_override(structure_1vme, sequence)


def test_multichain_override_with_ligands(structure_1vme: dict):
    """Both 1VME protein chains align to their overrides; ligand chains are left alone."""
    from atomworks.enums import ChainType
    from atomworks.io.utils.sequence import get_1_from_3_letter_code

    chain_info = structure_1vme["chain_info"]
    protein_chain_ids = [cid for cid, info in chain_info.items() if info["chain_type"].is_protein()]
    overrides = {
        "A": chain_info["A"]["processed_entity_canonical_sequence"] + "GGG",
        "B": chain_info["B"]["processed_entity_canonical_sequence"] + "AAAA",
    }
    assert protein_chain_ids == ["A", "B"]

    updated = apply_sequence_override(structure_1vme, f"{overrides['A']}:{overrides['B']}")

    arr = updated["asym_unit"]
    is_protein = np.isin(arr.chain_id, protein_chain_ids)
    assert np.all(arr.seq_idx[~is_protein] == -1)
    for chain_id, override in overrides.items():
        assert updated["chain_info"][chain_id]["processed_entity_canonical_sequence"] == override
        chain = arr[..., arr.chain_id == chain_id]
        observed = [
            get_1_from_3_letter_code(n, ChainType.POLYPEPTIDE_L, use_closest_canonical=True)
            for n in chain.res_name
        ]
        assert [override[i] for i in chain.seq_idx] == observed
    for chain_id, info in chain_info.items():
        if not info["chain_type"].is_protein():
            assert updated["chain_info"][chain_id] == info


@pytest.mark.parametrize("sequence", ["ACD::EFG", "ACD:", ":ACD"])
def test_override_with_empty_chain_sequence_raises(sequence: str):
    """An empty chain between separators is rejected rather than silently skipped."""
    with pytest.raises(ValueError, match="empty"):
        apply_sequence_override(_two_chain_structure("ACD", "EFG"), sequence)
