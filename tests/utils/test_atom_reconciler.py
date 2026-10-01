"""Tests for AtomReconciler."""

import pytest
import torch
from sampleworks.utils.atom_reconciler import AtomReconciler

from tests.utils.atom_array_builders import build_test_atom_array


class TestIdentity:
    """Null Object behavior — identity reconciler is a passthrough."""

    def test_identity_no_mismatch(self):
        rec = AtomReconciler.identity(5)
        assert not rec.has_mismatch
        assert rec.n_model == 5
        assert rec.n_struct == 5
        assert rec.n_common == 5

    def test_identity_align_is_full_alignment(self):
        rec = AtomReconciler.identity(4)
        model = torch.randn(2, 4, 3)
        ref = torch.randn(2, 4, 3)
        aligned, transform = rec.align(model, ref)
        assert aligned.shape == model.shape
        assert "rotation" in transform

    def test_identity_struct_to_model_is_copy(self):
        rec = AtomReconciler.identity(3)
        coords = torch.randn(2, 3, 3)
        template = torch.zeros(2, 3, 3)
        result = rec.struct_to_model(coords, template)
        torch.testing.assert_close(result, coords)

    def test_identity_model_to_struct_is_copy(self):
        rec = AtomReconciler.identity(3)
        coords = torch.randn(2, 3, 3)
        template = torch.zeros(2, 3, 3)
        result = rec.model_to_struct(coords, template)
        torch.testing.assert_close(result, coords)


class TestFromArrays:
    """Construction from model and structure atom arrays."""

    def test_matching_arrays_give_identity(self):
        arr = build_test_atom_array(chain_ids=["A", "A"], res_ids=[1, 1], atom_names=["N", "CA"])
        rec = AtomReconciler.from_arrays(arr, arr)
        assert not rec.has_mismatch

    def test_mismatched_arrays(self):
        model = build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[0, 0, 1], atom_names=["N", "CA", "N"]
        )
        struct = build_test_atom_array(chain_ids=["A", "A"], res_ids=[5, 6], atom_names=["N", "N"])
        rec = AtomReconciler.from_arrays(model, struct)
        assert rec.has_mismatch
        assert rec.n_model == 3
        assert rec.n_struct == 2
        assert rec.n_common == 2

    def test_no_common_atoms_raises(self):
        model = build_test_atom_array(chain_ids=["A", "A"], res_ids=[0, 1], atom_names=["N", "N"])
        struct = build_test_atom_array(chain_ids=["B"], res_ids=[0], atom_names=["CA"])
        with pytest.raises(RuntimeError, match="No common atoms"):
            AtomReconciler.from_arrays(model, struct)

    def test_different_chain_and_numbering_require_reconciliation(self):
        model = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "CA"]
        )
        struct = build_test_atom_array(
            chain_ids=["B", "B"], res_ids=[365, 365], atom_names=["N", "CA"]
        )

        rec = AtomReconciler.from_arrays(model, struct)

        assert rec.has_mismatch
        assert rec.n_common == len(model) == len(struct)

    def test_mse_se_maps_to_met_sd(self):
        model = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "SD"]
        )
        model.res_name[:] = "MET"
        model.element[:] = ["N", "S"]
        struct = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[13, 13], atom_names=["N", "SE"]
        )
        struct.res_name[:] = "MSE"
        struct.element[:] = ["N", "Se"]

        rec = AtomReconciler.from_arrays(model, struct)

        assert rec.has_mismatch
        assert rec.n_common == len(model) == len(struct)
        assert rec.model_indices.tolist() == [0, 1]
        assert rec.struct_indices.tolist() == [0, 1]


class TestCoordinateTranslation:
    """Round-trip and index correctness for struct → model mapping and alignment."""

    @pytest.fixture
    def mismatch_rec(self):
        """Model has 5 atoms [N,CA,CB,N,CA], struct has 3 [N,CA,N]. Common: N,CA,N."""
        model = build_test_atom_array(
            chain_ids=["A", "A", "A", "A", "A"],
            res_ids=[0, 0, 0, 1, 1],
            atom_names=["N", "CA", "CB", "N", "CA"],
        )
        struct = build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[5, 5, 6], atom_names=["N", "CA", "N"]
        )
        return AtomReconciler.from_arrays(model, struct)

    def test_struct_to_model_copies_common_atoms(self, mismatch_rec):
        rec = mismatch_rec
        struct_coords = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        template = torch.full((1, 5, 3), -1.0)
        result = rec.struct_to_model(struct_coords, template)
        assert result.shape == (1, 5, 3)
        for i in range(rec.n_common):
            m_i = int(rec.model_indices[i])
            s_i = int(rec.struct_indices[i])
            torch.testing.assert_close(result[0, m_i], struct_coords[0, s_i])
        # Non-common model atoms should retain template value
        common_model_set = set(rec.model_indices.tolist())
        for j in range(rec.n_model):
            if j not in common_model_set:
                torch.testing.assert_close(result[0, j], torch.tensor([-1.0, -1.0, -1.0]))

    def test_struct_to_model_rejects_wrong_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected struct_coords"):
            rec.struct_to_model(torch.randn(1, 4, 3), torch.randn(1, 5, 3))
        with pytest.raises(ValueError, match="Expected model_template"):
            rec.struct_to_model(torch.randn(1, 3, 3), torch.randn(1, 4, 3))

    def test_model_to_struct_copies_common_atoms(self, mismatch_rec):
        rec = mismatch_rec
        model_coords = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        template = torch.full((1, 3, 3), -1.0)

        result = rec.model_to_struct(model_coords, template)

        assert result.shape == template.shape
        torch.testing.assert_close(
            result[0, rec.struct_indices], model_coords[0, rec.model_indices]
        )

    def test_model_to_struct_rejects_wrong_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected model_coords"):
            rec.model_to_struct(torch.randn(1, 4, 3), torch.randn(1, 3, 3))
        with pytest.raises(ValueError, match="Expected struct_template"):
            rec.model_to_struct(torch.randn(1, 5, 3), torch.randn(1, 4, 3))

    def test_mapped_struct_mask_excludes_unmapped_template_atoms(self):
        model = build_test_atom_array(
            chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "CA"]
        )
        struct = build_test_atom_array(
            chain_ids=["B", "B", "B"], res_ids=[5, 5, 5], atom_names=["N", "CA", "O"]
        )
        rec = AtomReconciler.from_arrays(model, struct)

        mask = rec.mapped_struct_mask

        assert mask.dtype == torch.bool
        assert mask.tolist() == [True, True, False]

    def test_common_atoms_round_trip(self, mismatch_rec):
        rec = mismatch_rec
        struct_coords = torch.randn(2, 3, 3)
        model_template = torch.randn(2, 5, 3)
        struct_template = torch.randn(2, 3, 3)

        model_coords = rec.struct_to_model(struct_coords, model_template)
        result = rec.model_to_struct(model_coords, struct_template)

        torch.testing.assert_close(
            result[:, rec.struct_indices], struct_coords[:, rec.struct_indices]
        )

    def test_align_shape(self, mismatch_rec):
        rec = mismatch_rec
        model = torch.randn(2, 5, 3)
        model_ref = torch.randn(2, 5, 3)
        aligned, transform = rec.align(model, model_ref)
        assert aligned.shape == (2, 5, 3)
        assert "rotation" in transform

    def test_align_weights_accepted(self, mismatch_rec):
        rec = mismatch_rec
        model = torch.randn(1, 5, 3)
        model_ref = torch.randn(1, 5, 3)
        weights = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])  # (1, n_model=5)
        aligned, _ = rec.align(model, model_ref, align_weights=weights)
        assert aligned.shape == (1, 5, 3)

    def test_align_weights_influence_transform(self, mismatch_rec):
        """Different weights should produce different alignment transforms."""
        rec = mismatch_rec
        torch.manual_seed(42)
        model = torch.randn(1, 5, 3)
        model_ref = torch.randn(1, 5, 3)
        uniform = torch.ones(1, 5)
        sparse = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        aligned_uniform, _ = rec.align(model, model_ref, align_weights=uniform)
        aligned_sparse, _ = rec.align(model, model_ref, align_weights=sparse)
        assert not torch.allclose(aligned_uniform, aligned_sparse)

    def test_align_weights_rejects_wrong_size(self, mismatch_rec):
        rec = mismatch_rec
        model = torch.randn(1, 5, 3)
        model_ref = torch.randn(1, 5, 3)
        bad_weights = torch.ones(1, rec.n_common)  # n_common != n_model
        with pytest.raises(ValueError, match="align_weights last dimension must match n_model"):
            rec.align(model, model_ref, align_weights=bad_weights)

    def test_align_rejects_wrong_coord_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected model_coords"):
            rec.align(torch.randn(1, 3, 3), torch.randn(1, 5, 3))
        with pytest.raises(ValueError, match="Expected model_reference"):
            rec.align(torch.randn(1, 5, 3), torch.randn(1, 3, 3))

    def test_arbitrary_batch_dims(self, mismatch_rec):
        """struct_to_model handles extra leading batch dimensions."""
        rec = mismatch_rec
        struct = torch.randn(3, 4, 3, 3)
        template = torch.zeros(3, 4, 5, 3)
        result = rec.struct_to_model(struct, template)
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
        assert rec.model_indices.device.type == "cuda"
        assert rec.struct_indices.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_align_on_cuda(self):
        rec = AtomReconciler.identity(3).to("cuda")
        coords = torch.randn(1, 3, 3, device="cuda")
        ref = torch.randn(1, 3, 3, device="cuda")
        aligned, _ = rec.align(coords, ref)
        assert aligned.device.type == "cuda"
