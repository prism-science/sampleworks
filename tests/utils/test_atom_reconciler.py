"""Tests for AtomReconciler."""

import logging
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from sampleworks.utils.atom_reconciler import (
    _chain_residues,
    _fill_substitutions,
    _input_chain_seq_idx,
    _pair_input_chains,
    _place_input_residues,
    _sequence_identity,
    AtomReconciler,
)
from sampleworks.utils.frame_transforms import random_rotation_matrix

from tests.utils.atom_array_builders import build_test_atom_array


PREDICTED_NAMES = ["MET", "THR", "GLU", "ARG", "LYS", "GLY"]


def _chain_a(
    res_ids: Sequence[int], res_names: Sequence[str], atom_names: Sequence[str] = ("N", "CA")
) -> AtomArray:
    """Build chain A with one residue per ``res_ids`` entry, each holding ``atom_names``.

    Parameters
    ----------
    res_ids
        Residue number of each residue.
    res_names
        Residue name of each residue.
    atom_names
        Atoms of every residue, in order.

    Returns
    -------
    AtomArray
        ``len(res_ids) * len(atom_names)`` atoms, residue by residue.
    """
    chain = build_test_atom_array(
        chain_ids=["A"] * (len(res_ids) * len(atom_names)),
        res_ids=[res_id for res_id in res_ids for _ in atom_names],
        atom_names=list(atom_names) * len(res_ids),
    )
    chain.res_name[:] = [name for name in res_names for _ in atom_names]
    return chain


def _named_chains(chain_ids: list[str], names: dict[str, list[str]]) -> AtomArray:
    """Build two-residue chains with N and CA atoms, residues numbered 1 and 2.

    Parameters
    ----------
    chain_ids
        Chains in order.
    names
        Residue names of each chain.

    Returns
    -------
    AtomArray
        Four atoms per chain.
    """
    atoms = build_test_atom_array(
        chain_ids=[chain for chain in chain_ids for _ in range(4)],
        res_ids=[res_id for _ in chain_ids for res_id in (1, 1, 2, 2)],
        atom_names=["N", "CA"] * 2 * len(chain_ids),
    )
    atoms.res_name[:] = [name for chain in chain_ids for name in names[chain] for _ in "NC"]
    return atoms


def _pairs(rec: AtomReconciler) -> dict[int, int]:
    """Map each paired input atom index to its predicted atom index.

    Parameters
    ----------
    rec
        Reconciler to read the pairing from.

    Returns
    -------
    dict[int, int]
        Input atom index to predicted atom index.
    """
    return dict(zip(rec.input_indices.tolist(), rec.predicted_indices.tolist()))


class TestIdentity:
    """Null Object behavior — identity reconciler is a passthrough."""

    def test_identity_no_mismatch(self):
        rec = AtomReconciler.identity(5)
        assert not rec.has_mismatch
        assert rec.n_predicted == 5
        assert rec.n_input == 5
        assert rec.n_common == 5

    def test_identity_align_is_full_alignment(self):
        rec = AtomReconciler.identity(4)
        predicted_coords = torch.randn(2, 4, 3)
        ref = torch.randn(2, 4, 3)
        aligned, transform = rec.align(predicted_coords, ref)
        assert aligned.shape == predicted_coords.shape
        assert "rotation" in transform

    def test_identity_input_to_predicted_is_copy(self):
        rec = AtomReconciler.identity(3)
        coords = torch.randn(2, 3, 3)
        template = torch.zeros(2, 3, 3)
        result = rec.input_to_predicted(coords, template)
        torch.testing.assert_close(result, coords)


class TestAlignmentGradients:
    """Gradient behavior of align() on the identity reconciler."""

    def test_gradient_flow_through_alignment(self):
        """Gradients through alignment are rigid-motion invariant when allow_gradients=True.

        Aligned coordinates do not change under a rigid motion of the input, so the
        gradient of any function of them must exert zero net force and zero net
        torque on the input. A detached or partially detached transform breaks this.
        """
        torch.manual_seed(0)
        reconciler = AtomReconciler.identity(50)
        reference = torch.randn(1, 50, 3, dtype=torch.float64)
        rotation = random_rotation_matrix(reference)
        noise = 0.3 * torch.randn(reference.shape, dtype=reference.dtype)
        # (1, n_atoms, 3)
        predicted_coords = (
            (reference @ rotation.transpose(-1, -2) + noise).detach().requires_grad_(True)
        )
        # (1, n_atoms, 3), fixed per-coordinate weights so the objective is not symmetric
        objective_weights = torch.randn(reference.shape, dtype=reference.dtype)

        aligned, _ = reconciler.align(predicted_coords, reference, allow_gradients=True)
        assert aligned.requires_grad

        # (1, n_atoms, 3)
        grad = torch.autograd.grad((aligned * objective_weights).sum(), predicted_coords)[0]
        assert torch.isfinite(grad).all()
        grad_magnitude = grad.norm(dim=-1).sum()
        assert grad_magnitude > 0

        net_force = grad.sum(dim=-2)
        lever = predicted_coords.detach() - predicted_coords.detach().mean(dim=-2, keepdim=True)
        net_torque = torch.linalg.cross(lever, grad, dim=-1).sum(dim=-2)
        torque_magnitude = (lever.norm(dim=-1) * grad.norm(dim=-1)).sum()

        assert net_force.norm() / grad_magnitude < 1e-4
        assert net_torque.norm() / torque_magnitude < 1e-4

    def test_alignment_detaches_when_gradients_disallowed(self):
        """allow_gradients=False returns aligned coordinates with no autograd history."""
        reconciler = AtomReconciler.identity(20)
        reference = torch.randn(1, 20, 3)
        # The uniform +1.0 translation means alignment must undo a real transform rather than
        # return its input unchanged, so the no-grad check is meaningful.
        predicted_coords = (reference + 1.0).detach().requires_grad_(True)
        aligned, _ = reconciler.align(predicted_coords, reference, allow_gradients=False)

        assert not aligned.requires_grad


class TestFromArrays:
    """Construction from model and structure atom arrays."""

    def test_matching_arrays_give_identity(self):
        arr = build_test_atom_array(chain_ids=["A", "A"], res_ids=[1, 1], atom_names=["N", "CA"])
        rec = AtomReconciler.from_arrays(arr, arr)
        assert not rec.has_mismatch

    def test_mismatched_arrays(self):
        predicted_array = build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[0, 0, 1], atom_names=["N", "CA", "N"]
        )
        input_array = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[5, 6], atom_names=["N", "N"]
        )
        rec = AtomReconciler.from_arrays(predicted_array, input_array)
        assert rec.has_mismatch
        assert rec.n_predicted == 3
        assert rec.n_input == 2
        assert rec.n_common == 2

    def test_no_common_atoms_raises(self):
        predicted_array = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[0, 1], atom_names=["N", "N"]
        )
        input_array = build_test_atom_array(chain_ids=["B"], res_ids=[0], atom_names=["CA"])
        with pytest.raises(RuntimeError, match="No common atoms"):
            AtomReconciler.from_arrays(predicted_array, input_array)

    @pytest.mark.parametrize(
        "chain_ids,res_ids",
        [(["B"] * 4, [0, 0, 1, 1]), (["A"] * 4, [365, 365, 366, 366])],
        ids=["chain_only", "res_id_only"],
    )
    def test_label_only_difference_requires_reconciliation(self, chain_ids, res_ids):
        predicted_array = build_test_atom_array(
            chain_ids=["A"] * 4, res_ids=[0, 0, 1, 1], atom_names=["N", "CA"] * 2
        )
        input_array = build_test_atom_array(
            chain_ids=chain_ids, res_ids=res_ids, atom_names=["N", "CA"] * 2
        )

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert rec.n_common == len(predicted_array) == len(input_array)
        # Indices come back id-sorted, so compare them to each other rather than to a range.
        assert rec.predicted_indices.tolist() == rec.input_indices.tolist()
        assert rec.has_mismatch

    @pytest.mark.parametrize(
        "first_resolved", [3, 0], ids=["unmodeled_n_terminus", "unmodeled_c_terminus"]
    )
    def test_unresolved_terminus_pairs_same_residues(self, first_resolved):
        # Model built from the full sequence, but the structure lacks 3 terminal residues
        # and numbers the rest by sequence position
        predicted_array = _chain_a(range(6), PREDICTED_NAMES)
        resolved = range(first_resolved, first_resolved + 3)
        input_array = _chain_a([i + 1 for i in resolved], [PREDICTED_NAMES[i] for i in resolved])

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        # Two atoms per residue
        offset = 2 * first_resolved
        assert _pairs(rec) == {index: index + offset for index in range(len(input_array))}

    def test_differently_named_chains_pair_by_sequence(self):
        # The model lists B before A, so pairing L, H by order would put GLY-SER on TRP-TYR.
        predicted_array = _named_chains(["B", "A"], {"A": ["GLY", "SER"], "B": ["TRP", "TYR"]})
        input_array = _named_chains(["L", "H"], {"L": ["GLY", "SER"], "H": ["TRP", "TYR"]})

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert rec.n_common == 8
        assert (
            predicted_array.res_name[rec.predicted_indices.numpy()].tolist()
            == input_array.res_name[rec.input_indices.numpy()].tolist()
        )

    def test_identical_chains_pair_in_order_of_appearance(self):
        predicted_array = _named_chains(["A", "B"], {"A": ["GLY", "SER"], "B": ["GLY", "SER"]})
        input_array = _named_chains(["L", "H"], {"L": ["GLY", "SER"], "H": ["GLY", "SER"]})

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {index: index for index in range(len(input_array))}

    def test_chain_unlike_every_unpaired_chain_raises(self):
        predicted_array = _named_chains(["A", "B"], {"A": ["GLY", "SER"], "B": ["TRP", "TYR"]})
        input_array = _named_chains(["A", "X"], {"A": ["GLY", "SER"], "X": ["CYS", "CYS"]})

        with pytest.raises(ValueError, match="Input chain X is under 50% sequence identity"):
            AtomReconciler.from_arrays(predicted_array, input_array)

    def test_gap_in_repeated_residues_follows_input_numbering(self):
        # His-tag HHHH with the input missing residue 3: by name alone the gap could be anywhere.
        names = ["MET", "HIS", "HIS", "HIS", "HIS", "GLY"]
        predicted_array = _chain_a(range(1, 7), names, ["CA"])
        input_array = _chain_a([11, 12, 14, 15, 16], [names[i] for i in (0, 1, 3, 4, 5)], ["CA"])

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {0: 0, 1: 1, 2: 3, 3: 4, 4: 5}

    def test_mse_se_maps_to_met_sd(self):
        predicted_array = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "SD"]
        )
        predicted_array.res_name[:] = "MET"
        predicted_array.element[:] = ["N", "S"]
        input_array = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[13, 13], atom_names=["N", "SE"]
        )
        input_array.res_name[:] = "MSE"
        input_array.element[:] = ["N", "Se"]

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert rec.has_mismatch
        assert rec.n_common == len(predicted_array) == len(input_array)
        assert rec.predicted_indices.tolist() == [0, 1]
        assert rec.input_indices.tolist() == [0, 1]

    def test_point_mutation_pairs_as_in_a_full_length_chain(self, caplog):
        # ARG4ALA in an input missing residue 1: the backbone still belongs to model residue 4.
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        input_names = PREDICTED_NAMES[1:]
        input_names[2] = "ALA"
        input_array = _chain_a(range(2, 7), input_names)

        with caplog.at_level(logging.WARNING):
            rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {index: index + 2 for index in range(len(input_array))}
        assert caplog.text == ""

    def test_unmatched_terminal_residue_is_excluded_with_warning(self, caplog):
        # A terminal mismatch cannot be told from a residue the model lacks, so it pairs nothing.
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        input_names = PREDICTED_NAMES[1:]
        input_names[-1] = "TRP"
        input_array = _chain_a(range(2, 7), input_names)

        with caplog.at_level(logging.WARNING):
            rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {index: index + 2 for index in range(len(input_array) - 2)}
        assert "TRP6" in caplog.text

    @pytest.mark.parametrize("conformers", [("GLY", "ALA"), ("ALA", "GLY")])
    def test_microheterogeneity_matches_on_either_name(self, conformers):
        # Terminal residue 6 is GLY/ALA in the input, GLY in the model; residue 1 is unresolved.
        # Its first conformer pairs with the model residue whichever name comes first.
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        input_array = _chain_a([2, 3, 4, 5, 6, 6], [*PREDICTED_NAMES[1:5], *conformers])

        rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {index: index + 2 for index in range(len(input_array) - 2)}

    def test_matching_chain_logs_no_warning(self, caplog):
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)

        with caplog.at_level(logging.WARNING):
            AtomReconciler.from_arrays(predicted_array, _chain_a(range(11, 17), PREDICTED_NAMES))

        assert caplog.text == ""

    def test_equal_length_chain_with_tag_and_missing_terminus_pairs_same_residues(self, caplog):
        # A SER tag before MET and GLY6 unresolved: as many residues as the model, one off.
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        input_array = _chain_a(range(0, 6), ["SER", *PREDICTED_NAMES[:5]])

        with caplog.at_level(logging.WARNING):
            rec = AtomReconciler.from_arrays(predicted_array, input_array)

        assert _pairs(rec) == {index: index - 2 for index in range(2, len(input_array))}
        assert "SER0" in caplog.text

    @pytest.mark.parametrize("tag_first", [True, False], ids=["n_terminal_tag", "c_terminal_tag"])
    def test_longer_chain_skips_residues_the_prediction_lacks(self, tag_first, caplog):
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        names = ["SER", *PREDICTED_NAMES] if tag_first else [*PREDICTED_NAMES, "SER"]
        input_array = _chain_a(range(0, 7), names)

        with caplog.at_level(logging.WARNING):
            rec = AtomReconciler.from_arrays(predicted_array, input_array)

        offset = 2 if tag_first else 0
        assert _pairs(rec) == {index + offset: index for index in range(len(predicted_array))}
        assert "SER" in caplog.text

    def test_waters_in_a_protein_chain_warn(self, caplog):
        predicted_array = _chain_a(range(1, 7), PREDICTED_NAMES)
        input_array = _chain_a(range(2, 7), PREDICTED_NAMES[1:]) + _chain_a([101], ["HOH"], ["O"])

        with caplog.at_level(logging.WARNING):
            AtomReconciler.from_arrays(predicted_array, input_array)

        assert "mixes amino acids with other residues ['HOH101']" in caplog.text
        assert "match no predicted residue" not in caplog.text

    def test_ambiguous_placement_warns(self, caplog):
        # His-tag with one HIS unresolved and the input renumbered contiguously: the gap could
        # follow any HIS.
        his_tag = ["MET", "HIS", "HIS", "HIS", "HIS", "GLY"]
        predicted_array = _chain_a(range(1, 7), his_tag, ["CA"])
        input_array = _chain_a(range(1, 6), [*his_tag[:4], "GLY"], ["CA"])

        with caplog.at_level(logging.WARNING):
            AtomReconciler.from_arrays(predicted_array, input_array)

        assert "several equally good ways" in caplog.text


def test_fill_substitutions_fills_only_runs_that_fit():
    positions = np.array([-1, 0, -1, -1, 3, -1, 7, -1])

    np.testing.assert_array_equal(_fill_substitutions(positions), [-1, 0, 1, 2, 3, -1, 7, -1])


@pytest.mark.parametrize(
    ("input_names", "predicted_names", "identity"),
    [
        (["GLY", "SER", "TRP"], ["GLY", "SER", "TRP"], 1.0),
        (["SER", "TRP"], ["MET", "GLY", "SER", "TRP", "TYR"], 1.0),
        (["GLY", "SER", "TRP"], ["GLY", "ALA", "SER", "TRP"], 0.75),
        (["CYS", "CYS"], ["GLY", "SER"], 0.0),
    ],
    ids=["identical", "fragment", "one_gap", "unrelated"],
)
def test_sequence_identity_counts_only_the_matched_span(input_names, predicted_names, identity):
    input_sets = [frozenset([name]) for name in input_names]
    predicted_sets = [frozenset([name]) for name in predicted_names]

    assert _sequence_identity(input_sets, predicted_sets) == pytest.approx(identity)


class TestPairInputChains:
    """Input chains are renamed to the predicted chains whose sequences they match."""

    SEQUENCES = {"X": ["GLY", "SER"], "Y": ["TRP", "TYR"], "Z": ["CYS", "HIS"]}

    def _chains(self, sequence_of_chain: dict[str, str]) -> AtomArray:
        """Build two-residue chains from keys of ``SEQUENCES``.

        Parameters
        ----------
        sequence_of_chain
            Chain ID to its ``SEQUENCES`` key, in chain order.

        Returns
        -------
        AtomArray
        """
        names = {chain: self.SEQUENCES[seq] for chain, seq in sequence_of_chain.items()}
        return _named_chains(list(sequence_of_chain), names)

    def _renamed(self, atoms: AtomArray, chain_id: str) -> AtomArray:
        """Copy ``atoms`` into chain ``chain_id``.

        Parameters
        ----------
        atoms
            Atoms of one chain.
        chain_id
            New chain ID.

        Returns
        -------
        AtomArray
        """
        renamed = atoms.copy()
        renamed.chain_id[:] = chain_id
        return renamed

    def test_shifted_chain_ids_pair_by_sequence(self):
        predicted_array = self._chains({"A": "X", "B": "Y", "C": "Z"})
        input_array = self._chains({"B": "X", "C": "Y", "D": "Z"})

        paired = _pair_input_chains(predicted_array, input_array)

        np.testing.assert_array_equal(paired.chain_id, predicted_array.chain_id)

    def test_identical_copies_keep_their_ids(self):
        predicted_array = self._chains({"A": "X", "B": "X"})
        input_array = self._chains({"B": "X", "A": "X"})

        paired = _pair_input_chains(predicted_array, input_array)

        np.testing.assert_array_equal(paired.chain_id, input_array.chain_id)

    def test_peptide_pairs_with_the_chain_closest_to_its_length(self):
        # The peptide also matches a stretch of the protein exactly, and no chain IDs are
        # shared, so only residue count tells the two partners apart.
        protein = _chain_a(range(1, 7), PREDICTED_NAMES)
        peptide = _chain_a([1, 2], PREDICTED_NAMES[2:4])
        predicted_array = self._renamed(protein, "A") + self._renamed(peptide, "B")
        input_array = self._renamed(peptide, "Y") + self._renamed(protein, "X")

        paired = _pair_input_chains(predicted_array, input_array)

        np.testing.assert_array_equal(paired.chain_id, np.repeat(["B", "A"], [4, 12]))

    def test_unpaired_chain_cannot_take_a_paired_chain_id(self):
        # Input B holds waters, so predicted B goes to input C, whose sequence it shares.
        predicted_array = self._chains({"A": "X", "B": "Y"})
        input_array = _named_chains(
            ["A", "B", "C"], {"A": ["GLY", "SER"], "B": ["HOH", "HOH"], "C": ["TRP", "TYR"]}
        )

        paired = _pair_input_chains(predicted_array, input_array)

        np.testing.assert_array_equal(paired.chain_id, np.repeat(["A", "", "B"], 4))


class TestPlaceInputResidues:
    """Each input residue is matched to the index of a predicted residue, or -1."""

    def _place(self, res_ids, input_names, predicted_names=PREDICTED_NAMES):
        """Place a chain-A input against ``predicted_names``.

        Parameters
        ----------
        res_ids
            Residue number of each input residue.
        input_names
            Residue name of each input residue.
        predicted_names
            Residue names of the predicted chain.

        Returns
        -------
        tuple[np.ndarray, bool]
            The result of ``_place_input_residues``.
        """
        input_chain_array = _chain_a(res_ids, input_names)
        return _place_input_residues(
            _chain_residues(input_chain_array),
            input_chain_array,
            [frozenset([name]) for name in predicted_names],
        )

    def test_shorter_chain_is_placed_by_its_residue_numbers(self):
        # Residue 4 (ARG) is unresolved, so numbering puts LYS5 at index 4, not 3.
        input_names = [PREDICTED_NAMES[index] for index in (1, 2, 4, 5)]

        positions, ambiguous = self._place([2, 3, 5, 6], input_names)

        np.testing.assert_array_equal(positions, [1, 2, 4, 5])
        assert not ambiguous

    def test_longer_chain_skips_a_tag_the_prediction_lacks(self):
        positions, ambiguous = self._place(range(0, 7), ["SER", *PREDICTED_NAMES])

        np.testing.assert_array_equal(positions, [-1, 0, 1, 2, 3, 4, 5])
        assert not ambiguous

    def test_water_in_the_chain_is_not_aligned(self):
        # Not counting the water, the chain is shorter than the prediction, so its residue
        # numbers place the unresolved HIS4.
        his_tag = ["MET", "HIS", "HIS", "HIS", "HIS", "GLY"]

        positions, ambiguous = self._place(
            [1, 2, 3, 5, 6, 101], ["MET", "HIS", "HIS", "HIS", "GLY", "HOH"], his_tag
        )

        np.testing.assert_array_equal(positions, [0, 1, 2, 4, 5, -1])
        assert not ambiguous

    def test_gap_in_a_repeat_is_ambiguous(self):
        his_tag = ["MET", "HIS", "HIS", "HIS", "HIS", "GLY"]

        _, ambiguous = self._place(range(1, 6), [*his_tag[:4], "GLY"], his_tag)

        assert ambiguous


class TestInputChainSeqIdx:
    """Each input atom gets the index of its predicted residue."""

    def test_chain_matching_in_order_needs_no_indices(self):
        indices = _input_chain_seq_idx(
            "A", _chain_a(range(1, 7), PREDICTED_NAMES), _chain_a(range(11, 17), PREDICTED_NAMES)
        )

        np.testing.assert_array_equal(indices, -1)

    def test_unmatched_residue_gets_an_index_past_the_predicted_chain(self):
        input_names = PREDICTED_NAMES[1:]
        input_names[-1] = "TRP"

        indices = _input_chain_seq_idx(
            "A", _chain_a(range(1, 7), PREDICTED_NAMES), _chain_a(range(2, 7), input_names)
        )

        # Two atoms per residue. TRP6 matches no predicted residue, so it gets index 6, one
        # past the last predicted residue, which no predicted atom has.
        np.testing.assert_array_equal(indices, np.repeat([1, 2, 3, 4, 6], 2))


class TestCoordinateTranslation:
    """Round-trip and index correctness for struct → model mapping and alignment."""

    @pytest.fixture
    def mismatch_rec(self):
        """Model has 5 atoms [N,CA,CB,N,CA], struct has 3 [CA,N,N]. Common: N,CA,N.

        The struct lists residue 5 as CA then N, so pairing by atom position
        alone would cross N and CA.
        """
        predicted_array = build_test_atom_array(
            chain_ids=["A", "A", "A", "A", "A"],
            res_ids=[0, 0, 0, 1, 1],
            atom_names=["N", "CA", "CB", "N", "CA"],
        )
        input_array = build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[5, 5, 6], atom_names=["CA", "N", "N"]
        )
        return AtomReconciler.from_arrays(predicted_array, input_array)

    def test_input_to_predicted_copies_common_atoms(self, mismatch_rec):
        """Each struct atom lands on the model atom with the same residue and atom name."""
        rec = mismatch_rec
        # struct order: CA(res 5) -> [0,1,2], N(res 5) -> [3,4,5], N(res 6) -> [6,7,8]
        input_coords = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        template = torch.full((1, 5, 3), -1.0)
        result = rec.input_to_predicted(input_coords, template)
        # model order: N(res 0), CA(res 0), CB(res 0), N(res 1), CA(res 1)
        expected = torch.tensor(
            [
                [
                    [3.0, 4.0, 5.0],
                    [0.0, 1.0, 2.0],
                    [-1.0, -1.0, -1.0],
                    [6.0, 7.0, 8.0],
                    [-1.0, -1.0, -1.0],
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_input_to_predicted_rejects_wrong_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected input_coords"):
            rec.input_to_predicted(torch.randn(1, 4, 3), torch.randn(1, 5, 3))
        with pytest.raises(ValueError, match="Expected predicted_template"):
            rec.input_to_predicted(torch.randn(1, 3, 3), torch.randn(1, 4, 3))

    def test_align_shape(self, mismatch_rec):
        rec = mismatch_rec
        predicted_coords = torch.randn(2, 5, 3)
        predicted_reference = torch.randn(2, 5, 3)
        aligned, transform = rec.align(predicted_coords, predicted_reference)
        assert aligned.shape == (2, 5, 3)
        assert "rotation" in transform

    def test_align_weights_accepted(self, mismatch_rec):
        rec = mismatch_rec
        predicted_coords = torch.randn(1, 5, 3)
        predicted_reference = torch.randn(1, 5, 3)
        weights = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])  # (1, n_model=5)
        aligned, _ = rec.align(predicted_coords, predicted_reference, align_weights=weights)
        assert aligned.shape == (1, 5, 3)

    def test_align_weights_influence_transform(self, mismatch_rec):
        """Different weights should produce different alignment transforms."""
        rec = mismatch_rec
        torch.manual_seed(42)
        predicted_coords = torch.randn(1, 5, 3)
        predicted_reference = torch.randn(1, 5, 3)
        uniform = torch.ones(1, 5)
        sparse = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        aligned_uniform, _ = rec.align(predicted_coords, predicted_reference, align_weights=uniform)
        aligned_sparse, _ = rec.align(predicted_coords, predicted_reference, align_weights=sparse)
        assert not torch.allclose(aligned_uniform, aligned_sparse)

    def test_align_weights_rejects_wrong_size(self, mismatch_rec):
        rec = mismatch_rec
        predicted_coords = torch.randn(1, 5, 3)
        predicted_reference = torch.randn(1, 5, 3)
        bad_weights = torch.ones(1, rec.n_common)  # n_common != n_model
        with pytest.raises(ValueError, match="align_weights last dimension must match n_predicted"):
            rec.align(predicted_coords, predicted_reference, align_weights=bad_weights)

    def test_align_rejects_wrong_coord_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected predicted_coords"):
            rec.align(torch.randn(1, 3, 3), torch.randn(1, 5, 3))
        with pytest.raises(ValueError, match="Expected predicted_reference"):
            rec.align(torch.randn(1, 5, 3), torch.randn(1, 3, 3))

    def test_arbitrary_batch_dims(self, mismatch_rec):
        """input_to_predicted handles extra leading batch dimensions."""
        rec = mismatch_rec
        input_coords = torch.randn(3, 4, 3, 3)
        template = torch.zeros(3, 4, 5, 3)
        result = rec.input_to_predicted(input_coords, template)
        assert result.shape == (3, 4, 5, 3)


@pytest.mark.gpu
class TestDeviceHandling:
    """Reconciler.to() moves index tensors to match coordinate device."""

    def test_to_returns_self_when_already_on_device(self):
        rec = AtomReconciler.identity(3)
        assert rec.to("cpu") is rec

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self):
        rec = AtomReconciler.identity(3).to("cuda")
        assert rec.predicted_indices.device.type == "cuda"
        assert rec.input_indices.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_align_on_cuda(self):
        rec = AtomReconciler.identity(3).to("cuda")
        coords = torch.randn(1, 3, 3, device="cuda")
        ref = torch.randn(1, 3, 3, device="cuda")
        aligned, _ = rec.align(coords, ref)
        assert aligned.device.type == "cuda"
