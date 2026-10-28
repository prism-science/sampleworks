"""Tests for the BioEmu2 flow-model wrapper.

These use a randomly initialized BioEmu2 model (see ``conftest.py``), so they check
featurization, atom bookkeeping, batching and gradient flow, not sample quality.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.enums import ChainType
from bioemu2 import Ligand, Protein
from sampleworks.models.bioemu2.wrapper import (
    annotate_structure_for_bioemu2,
    BioEmu2Conditioning,
    BioEmu2Config,
    BioEmu2Wrapper,
    ligand_residues,
    resolve_msa,
    system_from_structure,
)
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.sequence import apply_sequence_override, expected_heavy_atom_count


def _offline(structure: dict, **kwargs) -> dict:
    """Annotate a structure so featurization never contacts the MSA server."""
    return annotate_structure_for_bioemu2(structure, use_msa_server=False, **kwargs)


def _conditioning(features: GenerativeModelInput[BioEmu2Conditioning]) -> BioEmu2Conditioning:
    assert features.conditioning is not None
    return features.conditioning


@pytest.fixture(scope="module")
def features_2yl0(
    tiny_bioemu2_wrapper: BioEmu2Wrapper, structure_2yl0_density: dict
) -> GenerativeModelInput[BioEmu2Conditioning]:
    return tiny_bioemu2_wrapper.featurize(_offline(structure_2yl0_density))


class TestSystemFromStructure:
    def test_protein_chain_uses_canonical_sequence(self, structure_2yl0_density: dict):
        system = system_from_structure(structure_2yl0_density)
        info = structure_2yl0_density["chain_info"]["A"]

        assert system.chain_ids == ["A"]
        (entity,) = system.entities
        assert isinstance(entity, Protein)
        assert entity.sequence == info["processed_entity_canonical_sequence"]

    def test_ligand_chains_follow_include_ligands(self, structure_1vme: dict):
        chain_info = structure_1vme["chain_info"]
        ligand_chains = {
            chain_id
            for chain_id, info in chain_info.items()
            if info["chain_type"] == ChainType.NON_POLYMER
        }
        assert ligand_chains, "1vme_final should carry ligand chains"

        with_ligands = system_from_structure(structure_1vme, include_ligands=True)
        without_ligands = system_from_structure(structure_1vme, include_ligands=False)

        ligand_ids = {
            chain_id
            for entity in with_ligands.entities
            if isinstance(entity, Ligand)
            for chain_id in entity.asym_ids
        }
        assert ligand_ids == ligand_chains
        assert set(without_ligands.chain_ids) == set(chain_info) - ligand_chains

    def test_multi_residue_ligand_chain_is_split(self, structure_2yl0: dict):
        """PDB files often put several hetero groups in one chain."""
        ligand_chains = {
            chain_id: list(info["res_name"])
            for chain_id, info in structure_2yl0["chain_info"].items()
            if info["chain_type"] == ChainType.NON_POLYMER and len(info["res_name"]) > 1
        }
        assert ligand_chains, "2YL0_single_001.pdb should have a multi-residue ligand chain"

        residues = ligand_residues(structure_2yl0)
        ligands = [
            e for e in system_from_structure(structure_2yl0).entities if isinstance(e, Ligand)
        ]
        assert len(ligands) == len(residues)
        for chain_id, res_names in ligand_chains.items():
            split = [ccd for input_chain, _, ccd in residues.values() if input_chain == chain_id]
            assert split == res_names
        assert all(len(asym_id) <= 4 for asym_id in residues)

    def test_sequence_override_carries_through(
        self, structure_5i09_density: dict, seq_5i09_deposited: str
    ):
        overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)
        (entity,) = system_from_structure(overridden).entities
        assert isinstance(entity, Protein)
        assert entity.sequence == seq_5i09_deposited

    def test_unsupported_chain_type_raises(self, structure_2yl0_density: dict):
        chain_info = {
            chain_id: {**info, "chain_type": ChainType.POLYPEPTIDE_D}
            for chain_id, info in structure_2yl0_density["chain_info"].items()
        }
        with pytest.raises(ValueError, match="does not support"):
            system_from_structure({**structure_2yl0_density, "chain_info": chain_info})


class TestResolveMSA:
    def test_offline_uses_single_sequence(self, structure_2yl0_density: dict):
        system = system_from_structure(structure_2yl0_density)
        msa = resolve_msa(system, BioEmu2Config(use_msa_server=False))
        sequence = structure_2yl0_density["chain_info"]["A"]["processed_entity_canonical_sequence"]
        assert msa.a3m("A") == f">query\n{sequence}\n"

    def test_a3m_path_is_loaded(self, structure_2yl0_density: dict, tmp_path: Path):
        system = system_from_structure(structure_2yl0_density)
        sequence = structure_2yl0_density["chain_info"]["A"]["processed_entity_canonical_sequence"]
        a3m = tmp_path / "query.a3m"
        a3m.write_text(f">query\n{sequence}\n>hit\n{sequence}\n")

        for msa_path in (a3m, {"A": str(a3m)}):
            msa = resolve_msa(system, BioEmu2Config(msa_path=msa_path))
            assert msa.a3m("A") == a3m.read_text()


class TestFeaturize:
    def test_implements_flow_model_wrapper(self, tiny_bioemu2_wrapper: BioEmu2Wrapper):
        assert isinstance(tiny_bioemu2_wrapper, FlowModelWrapper)

    def test_conditioning_describes_model_atoms(
        self, features_2yl0: GenerativeModelInput[BioEmu2Conditioning]
    ):
        cond = _conditioning(features_2yl0)

        assert cond.num_atoms == cond.topology.n_atoms == len(cond.model_atom_array)
        assert cond.num_padded_atoms % 32 == 0
        assert cond.num_padded_atoms >= cond.num_atoms
        assert cond.true_atom_array is not None

        topology_names = [atom.name for atom in cond.topology.atoms]
        np.testing.assert_array_equal(cond.model_atom_array.atom_name, topology_names)
        assert np.all(cond.model_atom_array.occupancy == 1.0)

    def test_input_atoms_reconcile_except_modified_residues_and_oxt(
        self, features_2yl0: GenerativeModelInput[BioEmu2Conditioning]
    ):
        """BioEmu2 models canonical residues without OXT; everything else must match."""
        cond = _conditioning(features_2yl0)
        model, struct = cond.model_atom_array, cond.true_atom_array
        reconciler = AtomReconciler.from_arrays(model, struct)

        unmatched = np.setdiff1d(np.arange(len(struct)), reconciler.struct_indices.numpy())
        model_res_names = dict(zip(model.res_id.tolist(), model.res_name.tolist()))
        for index in unmatched:
            is_oxt = struct.atom_name[index] == "OXT"
            in_modified_residue = model_res_names.get(int(struct.res_id[index])) != str(
                struct.res_name[index]
            )
            assert is_oxt or in_modified_residue, (
                f"unexpected unmatched atom {struct.res_name[index]}{struct.res_id[index]}:"
                f"{struct.atom_name[index]}"
            )

    def test_model_residue_ids_follow_input_numbering(
        self, features_2yl0: GenerativeModelInput[BioEmu2Conditioning], structure_2yl0_density: dict
    ):
        cond = _conditioning(features_2yl0)
        listed = [int(r) for r in structure_2yl0_density["chain_info"]["A"]["res_id"]]
        assert sorted(set(cond.model_atom_array.res_id.tolist())) == sorted(listed)

    def test_mode_is_recorded(
        self, tiny_bioemu2_wrapper: BioEmu2Wrapper, structure_2yl0_density: dict
    ):
        features = tiny_bioemu2_wrapper.featurize(
            _offline(structure_2yl0_density, mode="structure")
        )
        assert _conditioning(features).mode == "structure"

    def test_full_sequence_override_models_every_residue(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        structure_5i09_density: dict,
        seq_5i09_deposited: str,
    ):
        overridden = apply_sequence_override(structure_5i09_density, seq_5i09_deposited)
        cond = _conditioning(tiny_bioemu2_wrapper.featurize(_offline(overridden)))

        expected = expected_heavy_atom_count(seq_5i09_deposited)
        assert expected <= cond.num_atoms <= expected + 1  # +1 for a C-terminal OXT

        reconciler = AtomReconciler.from_arrays(cond.model_atom_array, cond.true_atom_array)
        assert reconciler.n_common > 0.95 * reconciler.n_struct


class TestStep:
    def test_shape_dtype_and_batch_sizes(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        num_atoms = _conditioning(features_2yl0).num_atoms
        for batch_size in (1, 3):
            x_t = 16.0 * tiny_bioemu2_wrapper.initialize_from_prior(batch_size, features_2yl0)
            x0 = tiny_bioemu2_wrapper.step(x_t, torch.tensor(16.0), features=features_2yl0)
            assert x0.shape == (batch_size, num_atoms, 3)
            assert x0.dtype == torch.float32
            assert torch.isfinite(x0).all()

    def test_samples_are_denoised_independently(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        """FK steering re-runs single particles; a sample must not depend on its batch."""
        x_t = 16.0 * tiny_bioemu2_wrapper.initialize_from_prior(3, features_2yl0)
        batched = tiny_bioemu2_wrapper.step(x_t, torch.tensor(8.0), features=features_2yl0)
        alone = tiny_bioemu2_wrapper.step(x_t[1:2], torch.tensor(8.0), features=features_2yl0)
        torch.testing.assert_close(batched[1:2], alone, rtol=1e-4, atol=1e-4)

    def test_scalar_and_per_sample_sigma_agree(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        x_t = 16.0 * tiny_bioemu2_wrapper.initialize_from_prior(2, features_2yl0)
        scalar = tiny_bioemu2_wrapper.step(x_t, 5.0, features=features_2yl0)
        per_sample = tiny_bioemu2_wrapper.step(
            x_t, torch.tensor([5.0, 5.0]), features=features_2yl0
        )
        torch.testing.assert_close(scalar, per_sample)

        mixed = tiny_bioemu2_wrapper.step(x_t, torch.tensor([5.0, 50.0]), features=features_2yl0)
        torch.testing.assert_close(mixed[:1], scalar[:1])
        assert not torch.allclose(mixed[1:], scalar[1:])

    def test_gradients_reach_coordinates(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        x_t = (16.0 * tiny_bioemu2_wrapper.initialize_from_prior(2, features_2yl0)).requires_grad_()
        tiny_bioemu2_wrapper.step(x_t, torch.tensor(4.0), features=features_2yl0).sum().backward()
        assert x_t.grad is not None
        assert torch.isfinite(x_t.grad).all()
        assert x_t.grad.abs().sum() > 0

    def test_gradients_reach_trunk_embeddings(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        """Latent optimization updates the trunk single and pair representations."""
        cond = _conditioning(features_2yl0)
        single = cond.single.detach().clone().requires_grad_()
        pair = cond.pair.detach().clone().requires_grad_()
        features = GenerativeModelInput(conditioning=replace(cond, single=single, pair=pair))

        x_t = 16.0 * tiny_bioemu2_wrapper.initialize_from_prior(1, features)
        tiny_bioemu2_wrapper.step(x_t, torch.tensor(4.0), features=features).sum().backward()
        for latent in (single, pair):
            assert latent.grad is not None
            assert latent.grad.abs().sum() > 0

    def test_replaced_pair_reaches_output_without_grad(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        """IT-opt's final pass samples with optimized latents under no_grad; they must count."""
        cond = _conditioning(features_2yl0)
        x_t = 16.0 * tiny_bioemu2_wrapper.initialize_from_prior(1, features_2yl0)
        perturbed = cond.pair + 3 * cond.pair.std() * torch.randn_like(cond.pair)
        with torch.no_grad():
            reference = tiny_bioemu2_wrapper.step(x_t, 4.0, features=features_2yl0)
            same = tiny_bioemu2_wrapper.step(
                x_t, 4.0, features=GenerativeModelInput(conditioning=replace(cond))
            )
            moved = tiny_bioemu2_wrapper.step(
                x_t, 4.0, features=GenerativeModelInput(conditioning=replace(cond, pair=perturbed))
            )
        torch.testing.assert_close(same, reference)
        assert not torch.allclose(moved, reference)

    def test_requires_features(self, tiny_bioemu2_wrapper: BioEmu2Wrapper):
        with pytest.raises(ValueError, match="features"):
            tiny_bioemu2_wrapper.step(torch.zeros(1, 4, 3), torch.tensor(1.0))


class TestInitializeFromPrior:
    def test_shape_from_features(
        self,
        tiny_bioemu2_wrapper: BioEmu2Wrapper,
        features_2yl0: GenerativeModelInput[BioEmu2Conditioning],
    ):
        prior = tiny_bioemu2_wrapper.initialize_from_prior(4, features_2yl0)
        assert prior.shape == (4, _conditioning(features_2yl0).num_atoms, 3)

    def test_explicit_shape(self, tiny_bioemu2_wrapper: BioEmu2Wrapper):
        assert tiny_bioemu2_wrapper.initialize_from_prior(2, shape=(10, 3)).shape == (2, 10, 3)
        with pytest.raises(ValueError, match="shape"):
            tiny_bioemu2_wrapper.initialize_from_prior(2, shape=(10, 4))

    def test_requires_features_or_shape(self, tiny_bioemu2_wrapper: BioEmu2Wrapper):
        with pytest.raises(ValueError, match="features or shape"):
            tiny_bioemu2_wrapper.initialize_from_prior(2)
