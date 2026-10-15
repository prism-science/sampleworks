"""Tests for frame transforms from sampleworks.utils.frame_transforms."""

import einx
import jax
import numpy as np
import pytest
import torch
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    apply_inverse_transform,
    create_random_transform,
    weighted_rigid_align_differentiable,
)
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("shape", [(5, 5, 5, 10, 3), (5, 6, 10, 3), (5, 10, 3)])
class TestFrameTransforms:
    """Test frame transform utilities used by PureGuidance."""

    def test_transform_invertibility_torch(self, shape: tuple[int, ...]):
        """Test that forward→inverse transform is identity."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = apply_inverse_transform(augmented, transform)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_invertibility_jax(self, shape: tuple[int, ...]):
        """Test that forward→inverse transform is identity."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = jax.numpy.asarray(apply_inverse_transform(augmented, transform))

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_preserves_shape_torch(self, shape: tuple[int, ...]):
        """Test that transforms preserve coordinate shapes."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        assert augmented.shape == coords.shape

        recovered = apply_inverse_transform(augmented, transform)
        assert recovered.shape == coords.shape

    def test_transform_preserves_shape_jax(self, shape: tuple[int, ...]):
        """Test that transforms preserve coordinate shapes."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        assert augmented.shape == coords.shape

        recovered = apply_inverse_transform(augmented, transform)
        assert recovered.shape == coords.shape

    def test_transform_only_rotation_torch(self, shape: tuple[int, ...]):
        """Test that rotation_only option works correctly."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

        # translate away from origin
        translation = torch.tensor([10.0, 20.0, 30.0])
        coords = coords + translation
        transform = create_random_transform(coords, center_before_rotation=False)

        # no centering w/o translation
        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)
        # check that coordinates are not at origin after rotation-only transform
        assert not torch.allclose(
            einx.mean("... n c -> ... c", recovered), torch.zeros_like(translation)
        )
        assert not torch.allclose(
            einx.mean("... n c -> ... c", augmented), torch.zeros_like(translation)
        )
        assert not torch.allclose(
            einx.mean("... n c -> ... c", coords), torch.zeros_like(translation)
        )

        # check that mixing rotation_only values doesn't recover original coords
        augmented = apply_forward_transform(coords, transform, rotation_only=False)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        assert not torch.allclose(coords, recovered)

        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=False)

        assert not torch.allclose(coords, recovered)

        # check that forward transform with rotation_only = False can be undone
        # by inverse transform with rotation_only = True and modification of the
        # translation
        augmented = apply_forward_transform(coords, transform, rotation_only=False)
        recovered_no_translation = apply_inverse_transform(augmented, transform, rotation_only=True)
        # R.T @ t
        translation = einx.dot(
            "... j i, ... j -> ... i", transform["rotation"], transform["translation"]
        )
        recovered = einx.subtract(
            "... n c, ... c -> ... n c", recovered_no_translation, translation
        )
        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_only_rotation_jax(self, shape: tuple[int, ...]):
        """Test that rotation_only option works correctly."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

        # translate away from origin
        translation = jax.numpy.array([10.0, 20.0, 30.0])
        coords = coords + translation
        transform = create_random_transform(coords, center_before_rotation=False, key=1)

        # no centering w/o translation
        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)
        # check that coordinates are not at origin after rotation-only transform
        assert not jax.numpy.allclose(
            jax.numpy.mean(recovered, axis=-2), jax.numpy.zeros_like(translation)
        )
        assert not jax.numpy.allclose(
            jax.numpy.mean(augmented, axis=-2), jax.numpy.zeros_like(translation)
        )
        assert not jax.numpy.allclose(
            jax.numpy.mean(coords, axis=-2), jax.numpy.zeros_like(translation)
        )

        # check that mixing rotation_only values doesn't recover original coords
        augmented = apply_forward_transform(coords, transform, rotation_only=False)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=True)

        assert not jax.numpy.allclose(coords, recovered)

        augmented = apply_forward_transform(coords, transform, rotation_only=True)
        recovered = apply_inverse_transform(augmented, transform, rotation_only=False)

        assert not jax.numpy.allclose(coords, recovered)

        # check that forward transform with rotation_only = False can be undone
        # by inverse transform with rotation_only = True and modification of the
        # translation
        augmented = apply_forward_transform(coords, transform, rotation_only=False)
        recovered_no_translation = apply_inverse_transform(augmented, transform, rotation_only=True)
        # R.T @ t
        translation = einx.dot(
            "... j i, ... j -> ... i", transform["rotation"], transform["translation"]
        )
        recovered = einx.subtract(
            "... n c, ... c -> ... n c", recovered_no_translation, translation
        )
        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_rotation_matrix_orthogonal_torch(self, shape: tuple[int, ...]):
        """Test that rotation matrix is orthogonal."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        R = transform["rotation"]
        batch_shape = shape[:-2]
        I = torch.eye(3, device=R.device, dtype=R.dtype)
        if batch_shape:
            I = I.expand(*batch_shape, 3, 3)

        RT = einx.rearrange("... i j -> ... j i", R)
        RRT = einx.dot("... i j, ... j k -> ... i k", R, RT)

        torch.testing.assert_close(RRT, I, rtol=1e-5, atol=1e-5)

    def test_transform_rotation_matrix_orthogonal_jax(self, shape: tuple[int, ...]):
        """Test that rotation matrix is orthogonal."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords)

        R = jax.numpy.asarray(transform["rotation"])
        batch_shape = shape[:-2]
        I = jax.numpy.eye(3, dtype=R.dtype)
        if batch_shape:
            I = jax.numpy.broadcast_to(I, (*batch_shape, 3, 3))

        RT = einx.rearrange("... i j -> ... j i", R)
        RRT = jax.numpy.asarray(einx.dot("... i j, ... j k -> ... i k", R, RT))

        assert jax.numpy.allclose(RRT, I, rtol=1e-5, atol=1e-5)

    def test_transform_rotation_matrix_determinant_torch(self, shape: tuple[int, ...]):
        """Test that rotation matrix has determinant 1."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        R = transform["rotation"]
        det = torch.det(R)

        torch.testing.assert_close(det, torch.ones(*shape[:-2], dtype=det.dtype))

    def test_transform_rotation_matrix_determinant_jax(self, shape: tuple[int, ...]):
        """Test that rotation matrix has determinant 1."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords)

        R = jax.numpy.asarray(transform["rotation"])
        det = jax.numpy.linalg.det(R)

        assert jax.numpy.allclose(det, 1.0, rtol=1e-5, atol=1e-5)

    def test_create_random_transform_with_centering_torch(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=True creates centered transform."""
        coords = torch.randn(*shape) + torch.tensor([10.0, 20.0, 30.0])
        transform = create_random_transform(coords, center_before_rotation=True)

        centroid = coords.mean(dim=-2)
        expected_t = -centroid

        torch.testing.assert_close(transform["translation"], expected_t, rtol=1e-5, atol=1e-5)

    def test_create_random_transform_with_centering_jax(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=True creates centered transform."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape) + jax.numpy.array(
            [10.0, 20.0, 30.0]
        )
        transform = create_random_transform(coords, center_before_rotation=True)

        centroid = jax.numpy.mean(coords, axis=-2)
        expected_t = -centroid

        assert jax.numpy.allclose(
            jax.numpy.asarray(transform["translation"]),
            expected_t,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_create_random_transform_without_centering_torch(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=False creates random translation."""
        coords = torch.randn(shape)
        transform = create_random_transform(coords, center_before_rotation=False)

        assert transform["translation"].shape == (*shape[:-2], 3)
        assert transform["rotation"].shape == (*shape[:-2], 3, 3)

    def test_undo_transform_far_from_origin_torch(self, shape: tuple[int, ...]):
        """Test that transform and its inverse work far from the origin."""
        coords = torch.randn(*shape) + torch.tensor([1e6, -1e6, 1e6])
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = apply_inverse_transform(augmented, transform)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_undo_transform_far_from_origin_jax(self, shape: tuple[int, ...]):
        """Test that transform and its inverse work far from the origin."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape) + jax.numpy.array([1e6, -1e6, 1e6])
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = jax.numpy.asarray(apply_inverse_transform(augmented, transform))

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_create_random_transform_without_centering_jax(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=False creates random translation."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords, center_before_rotation=False)

        assert transform["translation"].shape == (*shape[:-2], 3)
        assert transform["rotation"].shape == (*shape[:-2], 3, 3)

    def test_takes_in_different_backends(self, shape: tuple[int, ...]):
        """Test that function works with both torch and jax arrays."""
        # Torch
        coords_torch = torch.randn(*shape)
        transform_torch = create_random_transform(coords_torch)
        assert isinstance(transform_torch["rotation"], torch.Tensor)
        assert isinstance(transform_torch["translation"], torch.Tensor)

        # JAX
        coords_jax = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform_jax = create_random_transform(coords_jax)
        assert isinstance(transform_jax["rotation"], jax.numpy.ndarray)
        assert isinstance(transform_jax["translation"], jax.numpy.ndarray)


def _rigidly_moved_cloud(
    shape: tuple[int, int, int], seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a random 3D point cloud and a copy moved by a known proper rigid motion.

    Parameters
    ----------
    shape : tuple[int, int, int]
        ``(batch, n, 3)`` shape of the cloud.
    seed : int, optional
        Seed for the random generator, by default 0.

    Returns
    -------
    cloud : np.ndarray
        float32, shape (batch, n, 3). Full-rank random coordinates.
    moved : np.ndarray
        float32, shape (batch, n, 3). ``moved[b] = cloud[b] @ rotation[b].T + translation[b]``.
    rotation : np.ndarray
        float32, shape (batch, 3, 3). Proper rotations (det = +1).
    translation : np.ndarray
        float32, shape (batch, 3).
    """
    rng = np.random.default_rng(seed)
    batch_size = shape[0]
    cloud = rng.normal(scale=5.0, size=shape)
    rotation = Rotation.random(batch_size, random_state=rng).as_matrix()
    translation = rng.normal(scale=10.0, size=(batch_size, 3))
    moved = np.einsum("bij,bnj->bni", rotation, cloud) + translation[:, None, :]
    return (
        cloud.astype(np.float32),
        moved.astype(np.float32),
        rotation.astype(np.float32),
        translation.astype(np.float32),
    )


def _principal_axis_cloud(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Build centered point clouds whose principal axes are x > y > z.

    Each cloud's second-moment matrix is diagonal with strictly decreasing entries, so
    the proper rotation that best superimposes it onto its mirror image through the
    xy-plane is the identity (Kabsch flips the smallest singular direction).

    Parameters
    ----------
    shape : tuple[int, int, int]
        ``(batch, n, 3)`` shape of the cloud.
    seed : int, optional
        Seed for the random generator, by default 0.

    Returns
    -------
    np.ndarray
        float32, shape (batch, n, 3). Centered coordinates in each cloud's principal frame.
    """
    rng = np.random.default_rng(seed)
    clouds = rng.normal(size=shape) * np.array([6.0, 3.0, 1.0])
    clouds -= clouds.mean(axis=1, keepdims=True)
    for cloud in clouds:
        _, eigenvectors = np.linalg.eigh(cloud.T @ cloud)  # ascending eigenvalues
        cloud[:] = cloud @ eigenvectors[:, ::-1]
    return clouds.astype(np.float32)


@pytest.mark.parametrize("shape", [(1, 10, 3), (2, 50, 3), (15, 100, 3)])
class TestWeightedRigidAlign:
    """Test weighted rigid alignment against known rigid motions."""

    def test_recovers_known_rigid_motion_torch(self, shape: tuple[int, int, int]):
        """A cloud moved by a known rotation and translation is aligned back exactly."""
        cloud, moved, _, _ = _rigidly_moved_cloud(shape)
        ones = torch.ones(shape[:2])

        aligned = weighted_rigid_align_differentiable(
            torch.from_numpy(moved), torch.from_numpy(cloud), weights=ones, mask=ones
        )

        torch.testing.assert_close(aligned, torch.from_numpy(cloud), atol=1e-4, rtol=1e-4)

    def test_recovers_known_rigid_motion_jax(self, shape: tuple[int, int, int]):
        """A cloud moved by a known rotation and translation is aligned back exactly."""
        cloud, moved, _, _ = _rigidly_moved_cloud(shape)
        ones = jax.numpy.ones(shape[:2])

        aligned = weighted_rigid_align_differentiable(
            jax.numpy.asarray(moved), jax.numpy.asarray(cloud), weights=ones, mask=ones
        )

        np.testing.assert_allclose(np.asarray(aligned), cloud, atol=1e-4, rtol=1e-4)

    def test_returned_transform_is_inverse_of_known_motion_torch(self, shape: tuple[int, int, int]):
        """The returned transform is exactly x -> R.T @ (x - t) and reproduces the output."""
        cloud, moved, rotation, translation = _rigidly_moved_cloud(shape)
        ones = torch.ones(shape[:2])
        moved_torch = torch.from_numpy(moved)

        aligned, transform = weighted_rigid_align_differentiable(
            moved_torch, torch.from_numpy(cloud), weights=ones, mask=ones, return_transforms=True
        )

        inverse_rotation = rotation.transpose(0, 2, 1)
        inverse_translation = -np.einsum("bij,bj->bi", inverse_rotation, translation)
        torch.testing.assert_close(
            transform["rotation"], torch.from_numpy(inverse_rotation), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            transform["translation"], torch.from_numpy(inverse_translation), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            apply_forward_transform(moved_torch, transform), aligned, atol=1e-4, rtol=1e-4
        )

    def test_returned_transform_is_inverse_of_known_motion_jax(self, shape: tuple[int, int, int]):
        """The returned transform is exactly x -> R.T @ (x - t) and reproduces the output."""
        cloud, moved, rotation, translation = _rigidly_moved_cloud(shape)
        ones = jax.numpy.ones(shape[:2])
        moved_jax = jax.numpy.asarray(moved)

        aligned, transform = weighted_rigid_align_differentiable(
            moved_jax, jax.numpy.asarray(cloud), weights=ones, mask=ones, return_transforms=True
        )

        inverse_rotation = rotation.transpose(0, 2, 1)
        inverse_translation = -np.einsum("bij,bj->bi", inverse_rotation, translation)
        np.testing.assert_allclose(
            np.asarray(transform["rotation"]), inverse_rotation, atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(
            np.asarray(transform["translation"]), inverse_translation, atol=1e-4, rtol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(apply_forward_transform(moved_jax, transform)),
            np.asarray(aligned),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_mirror_image_is_matched_by_a_proper_rotation_torch(self, shape: tuple[int, int, int]):
        """Aligning onto a mirror image never reflects: the best proper rotation is used."""
        cloud = _principal_axis_cloud(shape)
        mirrored = cloud * np.array([1.0, 1.0, -1.0], dtype=np.float32)
        ones = torch.ones(shape[:2])

        aligned, transform = weighted_rigid_align_differentiable(
            torch.from_numpy(cloud),
            torch.from_numpy(mirrored),
            weights=ones,
            mask=ones,
            return_transforms=True,
        )

        torch.testing.assert_close(torch.linalg.det(transform["rotation"]), torch.ones(shape[0]))
        torch.testing.assert_close(aligned, torch.from_numpy(cloud), atol=1e-4, rtol=1e-4)

    def test_mirror_image_is_matched_by_a_proper_rotation_jax(self, shape: tuple[int, int, int]):
        """Aligning onto a mirror image never reflects: the best proper rotation is used."""
        cloud = _principal_axis_cloud(shape)
        mirrored = cloud * np.array([1.0, 1.0, -1.0], dtype=np.float32)
        ones = jax.numpy.ones(shape[:2])

        aligned, transform = weighted_rigid_align_differentiable(
            jax.numpy.asarray(cloud),
            jax.numpy.asarray(mirrored),
            weights=ones,
            mask=ones,
            return_transforms=True,
        )

        np.testing.assert_allclose(
            np.linalg.det(np.asarray(transform["rotation"])), np.ones(shape[0]), atol=1e-5
        )
        np.testing.assert_allclose(np.asarray(aligned), cloud, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("excluded_by", ["mask", "weights"])
    def test_excluded_atoms_do_not_affect_alignment_torch(
        self, shape: tuple[int, int, int], excluded_by: str
    ):
        """Reference atoms with zero mask or weight are ignored, however far off they are."""
        cloud, moved, _, _ = _rigidly_moved_cloud(shape)
        excluded = np.arange(shape[1]) >= shape[1] - shape[1] // 3
        corrupted_reference = cloud.copy()
        corrupted_reference[:, excluded] += 50.0
        included = torch.from_numpy(~excluded).float().expand(shape[0], -1)
        ones = torch.ones(shape[:2])
        weights, mask = (ones, included) if excluded_by == "mask" else (included, ones)

        aligned = weighted_rigid_align_differentiable(
            torch.from_numpy(moved),
            torch.from_numpy(corrupted_reference),
            weights=weights,
            mask=mask,
        )

        # Every atom, excluded ones included, lands on its true (uncorrupted) position.
        torch.testing.assert_close(aligned, torch.from_numpy(cloud), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("excluded_by", ["mask", "weights"])
    def test_excluded_atoms_do_not_affect_alignment_jax(
        self, shape: tuple[int, int, int], excluded_by: str
    ):
        """Reference atoms with zero mask or weight are ignored, however far off they are."""
        cloud, moved, _, _ = _rigidly_moved_cloud(shape)
        excluded = np.arange(shape[1]) >= shape[1] - shape[1] // 3
        corrupted_reference = cloud.copy()
        corrupted_reference[:, excluded] += 50.0
        included = jax.numpy.broadcast_to(
            jax.numpy.asarray(~excluded, dtype=jax.numpy.float32), shape[:2]
        )
        ones = jax.numpy.ones(shape[:2])
        weights, mask = (ones, included) if excluded_by == "mask" else (included, ones)

        aligned = weighted_rigid_align_differentiable(
            jax.numpy.asarray(moved),
            jax.numpy.asarray(corrupted_reference),
            weights=weights,
            mask=mask,
        )

        # Every atom, excluded ones included, lands on its true (uncorrupted) position.
        np.testing.assert_allclose(np.asarray(aligned), cloud, atol=1e-4, rtol=1e-4)

    def test_consistency_with_boltz(self, shape: tuple[int, ...]):
        """Test that PyTorch, JAX, and Boltz implementations are consistent."""
        boltz_diffusion = pytest.importorskip("boltz.model.loss.diffusion")
        batch_size = shape[0]
        num_points = shape[1]

        # Create test data in PyTorch
        torch.manual_seed(1)
        true_coords_torch = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
        pred_coords_torch = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
        weights_torch = torch.ones(batch_size, num_points, dtype=torch.float32)
        mask_torch = torch.ones(batch_size, num_points, dtype=torch.float32)

        # Run Boltz implementation (original)
        aligned_boltz = boltz_diffusion.weighted_rigid_align(
            true_coords_torch, pred_coords_torch, weights_torch, mask_torch
        )

        # Run our PyTorch implementation
        aligned_torch = weighted_rigid_align_differentiable(
            true_coords_torch,
            pred_coords_torch,
            weights_torch,
            mask_torch,
            allow_gradients=False,
        )

        # Convert to JAX and run JAX implementation
        true_coords_jax = jax.numpy.array(true_coords_torch.numpy())
        pred_coords_jax = jax.numpy.array(pred_coords_torch.numpy())
        weights_jax = jax.numpy.array(weights_torch.numpy())
        mask_jax = jax.numpy.array(mask_torch.numpy())

        aligned_jax = weighted_rigid_align_differentiable(
            true_coords_jax,
            pred_coords_jax,
            weights_jax,
            mask_jax,
            allow_gradients=False,
        )

        # Verify all three implementations match within tolerance
        torch.testing.assert_close(
            aligned_torch, aligned_boltz, rtol=5e-2, atol=1e-3, msg="PyTorch vs Boltz"
        )

        # Convert JAX array to numpy then to torch for comparison
        aligned_jax_np = np.asarray(aligned_jax)
        torch.testing.assert_close(
            torch.from_numpy(aligned_jax_np).clone(),
            aligned_boltz,
            rtol=5e-2,
            atol=1e-3,
            msg="JAX vs Boltz",
        )
