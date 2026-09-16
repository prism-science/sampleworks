"""Tests for the Protpardelle flow-model wrapper.

These tests build a small, randomly-initialized ``ai-allatom`` model (see
``conftest.py``) so they exercise real featurization/sampling logic without
needing downloaded weights.
"""

import atexit
import os
import shutil
import tempfile

import pytest


# Ensure the model-params directory exists before protpardelle is imported, mirroring
# conftest (import order across conftests is not guaranteed). Whichever module runs first
# creates the throwaway dir and registers its cleanup; the other sees the var already set
# and does nothing. Guard explicitly rather than `setdefault`, which would eagerly leak a
# mkdtemp even when the var is present.
if "PROTPARDELLE_MODEL_PARAMS" not in os.environ:
    _model_params_dir = tempfile.mkdtemp(prefix="protpardelle_model_params_")
    os.environ["PROTPARDELLE_MODEL_PARAMS"] = _model_params_dir
    atexit.register(shutil.rmtree, _model_params_dir, ignore_errors=True)

pytest.importorskip(
    "protpardelle.core.models", reason="Protpardelle not installed in this environment"
)

from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.io.parser import parse
from protpardelle.common import residue_constants
from protpardelle.data.sequence import seq_to_aatype
from sampleworks.models.protocol import GenerativeModelInput, StructureModelWrapper
from sampleworks.models.protpardelle.wrapper import (
    _atom37_indices_from_atom_array,
    _convert_atom37_to_flat,
    _convert_to_atom37,
    annotate_structure_for_protpardelle,
    ATOM37_ATOM_NAME_ALIASES,
    extract_protein_sequences,
    NUM_AATYPE_TOKENS,
    ProtpardelleConditioning,
    ProtpardelleConfig,
)
from sampleworks.utils.structure_utils import get_asym_unit_from_structure


SEQ_A = "ACDEFGHIKL"
SEQ_B = "MNPQRSTVWY"

# A real crystallographic structure (hen egg-white lysozyme) shipped as a test
# fixture; used to verify atom37 round-tripping preserves real atom ordering.
_REAL_CIF = (
    Path(__file__).resolve().parents[2] / "resources" / "1vme" / "1VME_single_001_density_input.cif"
)


def _build_asym_unit(sequences) -> struc.AtomArray:
    """Build a biotite AtomArray for the given chains.

    Each residue carries exactly the atom37 heavy atoms implied by its type
    (matching ``atom37_mask_from_aatype``), so the atom count agrees with the
    conditioning's ``atom_mask``. Coordinates are arbitrary but distinct.
    """
    chain_ids = "ABCDEFGH"
    atom_mask = np.asarray(residue_constants.restype_atom37_mask)
    atom_names, res_ids, chains = [], [], []
    for chain_idx, seq in enumerate(sequences):
        for res_pos, aa in enumerate(seq):
            restype = residue_constants.restype_order[aa]
            present = [
                residue_constants.atom_types[slot]
                for slot in range(residue_constants.atom_type_num)
                if atom_mask[restype][slot]
            ]
            atom_names.extend(present)
            res_ids.extend([res_pos + 1] * len(present))
            chains.extend([chain_ids[chain_idx]] * len(present))

    arr = struc.AtomArray(len(atom_names))
    arr.atom_name = np.array(atom_names)
    arr.res_id = np.array(res_ids)
    arr.chain_id = np.array(chains)
    arr.coord = np.arange(len(atom_names) * 3, dtype=np.float32).reshape(-1, 3)
    return arr


def _protein_structure(*sequences: str) -> dict:
    """Build a minimal structure dict with protein chain_info and an asym_unit."""
    chain_ids = "ABCDEFGH"
    chain_info = {
        chain_ids[i]: {
            "chain_type": ChainType.POLYPEPTIDE_L,
            "processed_entity_canonical_sequence": seq,
        }
        for i, seq in enumerate(sequences)
    }
    return {
        "chain_info": chain_info,
        "asym_unit": _build_asym_unit(sequences),
        "metadata": {"id": "test"},
    }


def _load_protein_heavy_atoms(cif_path: Path) -> struc.AtomArray:
    """Load a CIF and return its protein heavy atoms recognized by atom37.

    Parses the real fixture, drops waters / ligands / hydrogens and any atom
    whose (aliased) name is not an atom37 type, so the result is exactly what
    :func:`_atom37_indices_from_atom_array` accepts while preserving the file's
    original per-atom ordering (e.g. ``... C, O, CB ...``).
    """
    atom_array = get_asym_unit_from_structure(parse(str(cif_path), ccd_mirror_path=None), 0)
    aliased = np.array([ATOM37_ATOM_NAME_ALIASES.get(str(n), str(n)) for n in atom_array.atom_name])
    keep = np.isin(aliased, list(residue_constants.atom_order)) & struc.filter_amino_acids(
        atom_array
    )
    return atom_array[keep]


# ---------------------------------------------------------------------------
# Pure-logic tests (no model needed)
# ---------------------------------------------------------------------------


class TestExtractProteinSequences:
    def test_returns_sequences_in_chain_order(self):
        structure = _protein_structure(SEQ_A, SEQ_B)
        assert extract_protein_sequences(structure) == [SEQ_A, SEQ_B]

    def test_skips_non_protein_chains(self):
        structure = _protein_structure(SEQ_A)
        structure["chain_info"]["L"] = {
            "chain_type": ChainType.NON_POLYMER,
            "processed_entity_canonical_sequence": "",
        }
        assert extract_protein_sequences(structure) == [SEQ_A]

    def test_raises_without_chain_info(self):
        with pytest.raises(ValueError, match="chain_info"):
            extract_protein_sequences({"metadata": {"id": "x"}})

    def test_raises_without_protein_chains(self):
        structure = {
            "chain_info": {
                "L": {
                    "chain_type": ChainType.NON_POLYMER,
                    "processed_entity_canonical_sequence": "",
                }
            }
        }
        with pytest.raises(ValueError, match="protein"):
            extract_protein_sequences(structure)


class TestAnnotateStructure:
    def test_adds_config(self):
        structure = _protein_structure(SEQ_A)
        annotated = annotate_structure_for_protpardelle(structure)
        config = annotated["_protpardelle_config"]
        assert isinstance(config, ProtpardelleConfig)
        # Original structure is not mutated.
        assert "_protpardelle_config" not in structure


class TestProtocolConformance:
    def test_is_structure_model_wrapper(self, protpardelle_wrapper):
        assert isinstance(protpardelle_wrapper, StructureModelWrapper)


# ---------------------------------------------------------------------------
# Featurization tests (small random model)
# ---------------------------------------------------------------------------


class TestFeaturize:
    def test_returns_generative_model_input(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        features = protpardelle_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert isinstance(features.conditioning, ProtpardelleConditioning)

    def test_conditioning_is_batch_one(self, protpardelle_wrapper):
        # Conditioning carries a single (batch=1) ensemble dim; the sampler owns
        # the ensemble size and step broadcasts to it.
        structure = annotate_structure_for_protpardelle(_protein_structure(SEQ_A))
        cond = protpardelle_wrapper.featurize(structure).conditioning
        length = len(SEQ_A)
        assert cond.aatype.shape == (1, length)
        assert cond.seq_mask.shape == (1, length)
        assert cond.residue_index.shape == (1, length)
        assert cond.atom_mask.shape == (1, length, 37)
        assert cond.sequences == (SEQ_A,)

    def test_aatype_matches_sequence(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        expected = seq_to_aatype(SEQ_A, num_tokens=NUM_AATYPE_TOKENS)
        assert torch.equal(cond.aatype[0].cpu(), expected)

    def test_atom37_index_maps_present(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        n_atoms = int(cond.atom_mask[0].sum().item())
        # One entry per flat atom, matching the atom37 occupancy count.
        assert cond.atom37_residue_index.shape == (n_atoms,)
        assert cond.atom37_atom_index.shape == (n_atoms,)
        # Slots are valid atom37 indices; residues stay within L.
        assert int(cond.atom37_atom_index.max()) < 37
        assert int(cond.atom37_residue_index.max()) == len(SEQ_A) - 1

    def test_mse_selenium_maps_to_methionine_sd(self, protpardelle_wrapper):
        structure = _protein_structure("M")
        atom_array = structure["asym_unit"]
        selenium_mask = atom_array.atom_name == "SD"
        atom_array.atom_name[selenium_mask] = "SE"

        cond = protpardelle_wrapper.featurize(structure).conditioning
        selenium_index = int(np.flatnonzero(selenium_mask)[0])

        assert cond.atom37_atom_index[selenium_index].item() == residue_constants.atom_order["SD"]

    def test_featurize_requires_asym_unit(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        del structure["asym_unit"]
        with pytest.raises(ValueError, match="asym_unit"):
            protpardelle_wrapper.featurize(structure)

    def test_multichain_indices_have_gap(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A, SEQ_B)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        # Two distinct chains present.
        chain_ids = torch.unique(cond.chain_index[0])
        assert chain_ids.numel() == 2
        # The residue index jumps by more than 1 at the chain boundary.
        residue_index = cond.residue_index[0]
        boundary_gap = residue_index[len(SEQ_A)] - residue_index[len(SEQ_A) - 1]
        assert boundary_gap > 1

    def test_featurize_filters_non_protein_atoms(self, protpardelle_wrapper):
        """Ligand / non-protein atoms in the asym_unit must not enter the atom37 mapping."""
        structure = _protein_structure(SEQ_A)

        # Append a ligand chain whose atom names are absent from the atom37
        # alphabet; if they reached the atom37 mapping they would raise, and
        # counting their residues would misalign the layout with the sequence.
        ligand = struc.AtomArray(2)
        ligand.atom_name = np.array(["ZN", "MG"])
        ligand.res_id = np.array([1, 2])
        ligand.chain_id = np.array(["L", "L"])
        ligand.coord = np.zeros((2, 3), dtype=np.float32)
        structure["asym_unit"] = structure["asym_unit"] + ligand
        structure["chain_info"]["L"] = {
            "chain_type": ChainType.NON_POLYMER,
            "processed_entity_canonical_sequence": "",
        }

        features = protpardelle_wrapper.featurize(structure)

        # The resulting layout matches the protein-only structure: the ligand
        # atoms were filtered out before the atom37 mapping (same flat atom count
        # and same residue span).
        protein_only = protpardelle_wrapper.featurize(_protein_structure(SEQ_A))
        assert (
            features.conditioning.atom37_residue_index.shape
            == protein_only.conditioning.atom37_residue_index.shape
        )
        assert int(features.conditioning.atom37_residue_index.max()) == len(SEQ_A) - 1


class TestInitializeFromPrior:
    def test_with_shape(self, protpardelle_wrapper):
        out = protpardelle_wrapper.initialize_from_prior(2, shape=(40, 3))
        assert out.shape == (2, 40, 3)

    def test_with_features(self, protpardelle_wrapper):
        features = protpardelle_wrapper.featurize(_protein_structure(SEQ_A))
        out = protpardelle_wrapper.initialize_from_prior(3, features=features)
        assert out.shape[0] == 3
        assert out.shape[-1] == 3

    def test_raises_without_features_or_shape(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="features|shape"):
            protpardelle_wrapper.initialize_from_prior(batch_size=2)

    def test_invalid_shape_raises(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="num_atoms"):
            protpardelle_wrapper.initialize_from_prior(1, shape=(40, 4))


# ---------------------------------------------------------------------------
# Step / sampling tests (small random model, short trajectory)
# ---------------------------------------------------------------------------


class TestConvertToAtom37:
    def test_shape_and_placement(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n_atoms = res_idx.shape[0]
        num_residues = int(res_idx.max()) + 1

        x_t = torch.randn(2, n_atoms, 3)
        out = _convert_to_atom37(x_t, res_idx, atom_idx, num_residues)

        assert out.shape == (2, num_residues, 37, 3)
        # Every flat atom lands at its (residue, slot) destination.
        for n in range(n_atoms):
            assert torch.allclose(out[:, res_idx[n], atom_idx[n]], x_t[:, n])
        # Total non-zero atom slots equals the number of placed atoms.
        occupied = (out.abs().sum(-1) > 0).sum().item()
        assert occupied == 2 * n_atoms

    def test_gradients_flow_back_to_x_t(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n_atoms = res_idx.shape[0]
        num_residues = int(res_idx.max()) + 1

        x_t = torch.randn(1, n_atoms, 3, requires_grad=True)
        out = _convert_to_atom37(x_t, res_idx, atom_idx, num_residues)
        out.sum().backward()
        # Each input coordinate is placed exactly once -> unit gradient.
        assert torch.allclose(x_t.grad, torch.ones_like(x_t))

    def test_atom_count_mismatch_raises(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        num_residues = int(res_idx.max()) + 1
        with pytest.raises(ValueError, match="atoms"):
            _convert_to_atom37(torch.randn(1, 3, 3), res_idx, atom_idx, num_residues)


@pytest.fixture(scope="module")
def real_atom_array() -> struc.AtomArray:
    """Protein heavy atoms from a real crystallographic CIF fixture (1VME)."""
    return _load_protein_heavy_atoms(_REAL_CIF)


class TestAtom37IndicesFromAtomArray:
    """Unit tests for the standalone ``_atom37_indices_from_atom_array``."""

    def test_slots_match_atom_names(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n = real_atom_array.array_length()
        assert res_idx.shape == (n,)
        assert atom_idx.shape == (n,)
        # Each atom37 slot is exactly residue_constants.atom_order[name].
        aliased = [
            ATOM37_ATOM_NAME_ALIASES.get(str(name), str(name)) for name in real_atom_array.atom_name
        ]
        expected = torch.as_tensor(
            [residue_constants.atom_order[a] for a in aliased], dtype=torch.long
        )
        assert torch.equal(atom_idx.cpu(), expected)

    def test_residue_ordinals_are_contiguous(self, real_atom_array):
        res_idx, _ = _atom37_indices_from_atom_array(real_atom_array)
        n_residues = len(struc.get_residue_starts(real_atom_array))
        assert int(res_idx[0]) == 0
        assert int(res_idx.max()) == n_residues - 1
        # Ordinals only ever stay the same or step up by one, in atom order.
        steps = res_idx[1:] - res_idx[:-1]
        assert torch.all(steps >= 0)
        assert set(steps.tolist()) <= {0, 1}

    def test_respects_device_argument(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array, device="cpu")
        assert res_idx.device.type == "cpu"
        assert atom_idx.device.type == "cpu"

    def test_unknown_atom_name_raises(self):
        arr = struc.AtomArray(1)
        arr.atom_name = np.array(["ZZ"])
        arr.res_id = np.array([1])
        arr.chain_id = np.array(["A"])
        arr.coord = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="atom37"):
            _atom37_indices_from_atom_array(arr)


class TestAtom37RoundTrip:
    """``flat -> atom37 -> flat`` must preserve the original atom order."""

    def test_roundtrip_preserves_order(self, real_atom_array):
        # Give every atom a unique sentinel coordinate equal to its index, so any
        # reordering during the round trip is immediately detectable.
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n = res_idx.shape[0]
        num_residues = int(res_idx.max()) + 1
        x_flat = torch.arange(n, dtype=torch.float64).reshape(1, n, 1).expand(1, n, 3).contiguous()

        x_atom37 = _convert_to_atom37(x_flat, res_idx, atom_idx, num_residues)
        recovered = _convert_atom37_to_flat(x_atom37, res_idx, atom_idx)

        assert recovered.shape == x_flat.shape
        assert torch.equal(recovered, x_flat)

    def test_roundtrip_preserves_real_coordinates(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        num_residues = int(res_idx.max()) + 1
        x_flat = torch.as_tensor(real_atom_array.coord, dtype=torch.float64)[None]

        x_atom37 = _convert_to_atom37(x_flat, res_idx, atom_idx, num_residues)
        recovered = _convert_atom37_to_flat(x_atom37, res_idx, atom_idx)

        # equal_nan: the fixture has a few unresolved (NaN) coordinates.
        assert torch.allclose(recovered, x_flat, equal_nan=True)

    def test_naive_mask_gather_would_reorder(self, real_atom_array):
        # Guards the actual bug: a boolean-mask gather emits atoms in atom37-slot
        # order, which differs from the input order (e.g. CIF stores O before CB
        # but slot CB=3 precedes O=4). The index-map gather must NOT do that.
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n = res_idx.shape[0]
        num_residues = int(res_idx.max()) + 1
        x_flat = torch.arange(n, dtype=torch.float64).reshape(1, n, 1).expand(1, n, 3).contiguous()
        x_atom37 = _convert_to_atom37(x_flat, res_idx, atom_idx, num_residues)

        ordered = _convert_atom37_to_flat(x_atom37, res_idx, atom_idx)
        mask = torch.zeros(num_residues, 37, dtype=torch.bool)
        mask[res_idx, atom_idx] = True
        naive = x_atom37[:, mask]

        # Same set of atoms, but the mask gather reorders them...
        assert naive.shape == ordered.shape
        assert not torch.equal(naive, x_flat)
        # ...while the index-map gather reproduces the input exactly.
        assert torch.equal(ordered, x_flat)

    def test_convert_atom37_to_flat_is_differentiable(self, real_atom_array):
        res_idx, atom_idx = _atom37_indices_from_atom_array(real_atom_array)
        n = res_idx.shape[0]
        num_residues = int(res_idx.max()) + 1

        x_atom37 = torch.randn(1, num_residues, 37, 3, dtype=torch.float64, requires_grad=True)
        out = _convert_atom37_to_flat(x_atom37, res_idx, atom_idx)
        out.sum().backward()

        # Each atom is gathered from a distinct slot exactly once -> unit gradient
        # at gathered slots, zero elsewhere.
        assert torch.equal(
            x_atom37.grad[0, res_idx, atom_idx], torch.ones(n, 3, dtype=torch.float64)
        )
        assert x_atom37.grad.sum().item() == n * 3


class TestStep:
    def test_step_raises_without_features(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="features"):
            protpardelle_wrapper.step(torch.zeros(1, 1, 3), 0.0)

    @pytest.mark.slow
    def test_step_returns_coords(self, protpardelle_wrapper):
        short_seq = "ACDEFG"
        structure = annotate_structure_for_protpardelle(_protein_structure(short_seq))
        features = protpardelle_wrapper.featurize(structure)
        # Ensemble batch (>1) is set by the sampler on x_t, not by featurization;
        # step must broadcast the batch-1 conditioning to match.
        ensemble_size = 3
        x_init = protpardelle_wrapper.initialize_from_prior(
            batch_size=ensemble_size, features=features
        )
        assert x_init.shape[0] == ensemble_size
        result = protpardelle_wrapper.step(x_init, 1.0, features=features)
        assert torch.is_tensor(result)
        assert result.shape == x_init.shape
        assert result.shape[-1] == 3
        assert torch.isfinite(result).all()
