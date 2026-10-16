"""Tests for AtomReconciler."""

import pytest
import torch
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.frame_transforms import random_rotation_matrix

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
        model_coords = (
            (reference @ rotation.transpose(-1, -2) + noise).detach().requires_grad_(True)
        )
        # (1, n_atoms, 3), fixed per-coordinate weights so the objective is not symmetric
        objective_weights = torch.randn(reference.shape, dtype=reference.dtype)

        aligned, _ = reconciler.align(model_coords, reference, allow_gradients=True)
        assert aligned.requires_grad

        # (1, n_atoms, 3)
        grad = torch.autograd.grad((aligned * objective_weights).sum(), model_coords)[0]
        assert torch.isfinite(grad).all()
        grad_magnitude = grad.norm(dim=-1).sum()
        assert grad_magnitude > 0

        net_force = grad.sum(dim=-2)
        lever = model_coords.detach() - model_coords.detach().mean(dim=-2, keepdim=True)
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
        model_coords = (reference + 1.0).detach().requires_grad_(True)
        aligned, _ = reconciler.align(model_coords, reference, allow_gradients=False)

        assert not aligned.requires_grad


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

    @pytest.mark.parametrize(
        "chain_ids,res_ids",
        [(["B"] * 4, [0, 0, 1, 1]), (["A"] * 4, [365, 365, 366, 366])],
        ids=["chain_only", "res_id_only"],
    )
    def test_label_only_difference_requires_reconciliation(self, chain_ids, res_ids):
        model = build_test_atom_array(
            chain_ids=["A"] * 4, res_ids=[0, 0, 1, 1], atom_names=["N", "CA"] * 2
        )
        struct = build_test_atom_array(
            chain_ids=chain_ids, res_ids=res_ids, atom_names=["N", "CA"] * 2
        )

        rec = AtomReconciler.from_arrays(model, struct)

        assert rec.n_common == len(model) == len(struct)
        # Indices come back id-sorted, so compare them to each other rather than to a range.
        assert rec.model_indices.tolist() == rec.struct_indices.tolist()
        assert rec.has_mismatch

    def test_mse_se_maps_to_met_sd(self):
        model = build_test_atom_array(chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "SD"])
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
        """Model has 5 atoms [N,CA,CB,N,CA], struct has 3 [CA,N,N]. Common: N,CA,N.

        The struct lists residue 5 as CA then N, so pairing by atom position
        alone would cross N and CA.
        """
        model = build_test_atom_array(
            chain_ids=["A", "A", "A", "A", "A"],
            res_ids=[0, 0, 0, 1, 1],
            atom_names=["N", "CA", "CB", "N", "CA"],
        )
        struct = build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[5, 5, 6], atom_names=["CA", "N", "N"]
        )
        return AtomReconciler.from_arrays(model, struct)

    def test_struct_to_model_copies_common_atoms(self, mismatch_rec):
        """Each struct atom lands on the model atom with the same residue and atom name."""
        rec = mismatch_rec
        # struct order: CA(res 5) -> [0,1,2], N(res 5) -> [3,4,5], N(res 6) -> [6,7,8]
        struct_coords = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        template = torch.full((1, 5, 3), -1.0)
        result = rec.struct_to_model(struct_coords, template)
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

    def test_struct_to_model_rejects_wrong_shapes(self, mismatch_rec):
        rec = mismatch_rec
        with pytest.raises(ValueError, match="Expected struct_coords"):
            rec.struct_to_model(torch.randn(1, 4, 3), torch.randn(1, 5, 3))
        with pytest.raises(ValueError, match="Expected model_template"):
            rec.struct_to_model(torch.randn(1, 3, 3), torch.randn(1, 4, 3))

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
